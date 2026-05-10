use async_stream::try_stream;
use axum::{
    BoxError, Json,
    response::{IntoResponse, Sse, sse::Event as SseEvent},
};
use bytes::Bytes;
use eventsource_stream::{EventStream, Eventsource};
use futures::{Stream, TryStreamExt};
use serde::Deserialize;
use tracing::debug;
use url::Url;
use wreq::Proxy;

use crate::{
    claude_code_state::ClaudeCodeState,
    claude_web_state::ClaudeWebState,
    error::{CheckClaudeErr, ClewdrError},
    services::tool_call::TOOL_CALL_MANAGER,
    types::claude::{
        ContentBlock, CountMessageTokensResponse, CreateMessageParams, CreateMessageResponse,
        Message, Role,
    },
    utils::print_out_text,
};

/// Merges server-sent events (SSE) from a stream into a single string
/// Extracts and concatenates completion data from events
///
/// # Arguments
/// * `stream` - Event stream to process
///
/// # Returns
/// Combined completion text from all events
pub async fn merge_sse(
    stream: EventStream<impl Stream<Item = Result<Bytes, wreq::Error>>>,
) -> Result<String, ClewdrError> {
    #[derive(Deserialize)]
    struct Data {
        completion: String,
    }
    Ok(stream
        .try_filter_map(async |event| {
            Ok(serde_json::from_str::<Data>(&event.data)
                .map(|data| data.completion)
                .ok())
        })
        .try_collect()
        .await?)
}

impl<S> From<S> for Message
where
    S: Into<String>,
{
    /// Converts a string into a Message with assistant role
    ///
    /// # Arguments
    /// * `str` - The text content for the message
    ///
    /// # Returns
    /// * `Message` - A message with assistant role and text content
    fn from(str: S) -> Self {
        Message::new_blocks(Role::Assistant, vec![ContentBlock::text(str.into())])
    }
}

impl ClaudeWebState {
    /// Converts the response from the Claude Web into Claude API or OpenAI API format
    ///
    /// This method transforms streams of bytes from Claude's web response into the appropriate
    /// format based on the client's requested API format (Claude or OpenAI). It handles both
    /// streaming and non-streaming responses, and manages caching for responses.
    ///
    /// # Arguments
    /// * `input` - The response stream from the Claude Web API
    ///
    /// # Returns
    /// * `axum::response::Response` - Transformed response in the requested format
    pub async fn transform_response(
        &mut self,
        wreq_res: wreq::Response,
    ) -> Result<axum::response::Response, ClewdrError> {
        if self.stream {
            // Stream through while accumulating completion text; persist usage at end
            let mut input_tokens = self.usage.input_tokens as u64;
            let handle = self.cookie_actor_handle.clone();
            let cookie = self.cookie.clone();
            let enable_precise = crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens;
            let last_params = self.last_params.clone();
            let endpoint = self.endpoint.clone();
            let proxy = self.proxy.clone();
            let client = self.client.clone();
            let has_tools = self
                .last_params
                .as_ref()
                .is_some_and(|p| p.tools.as_ref().is_some_and(|t| !t.is_empty()));
            let tool_call_session = self.clone();
            // try to get precise input tokens via Claude Code count_tokens if enabled
            if crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens
                && let Some(tokens) = self.try_code_count_tokens().await
            {
                input_tokens = tokens as u64;
            }

            let stream = wreq_res
                .bytes_stream()
                .eventsource()
                .map_err(axum::Error::new);
            let stream = try_stream! {
                let mut acc = String::new();
                #[derive(serde::Deserialize)]
                struct Data { completion: String }
                futures::pin_mut!(stream);

                // Tool call detection state
                let mut tool_use_id: Option<String> = None;
                let mut tool_use_index: Option<usize> = None;
                let mut tool_call_break = false;
                // Indices of server-side (built-in) tool blocks to suppress
                let mut suppressed_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

                while let Some(event) = stream.try_next().await? {
                    if let Ok(d) = serde_json::from_str::<Data>(&event.data) {
                        acc.push_str(&d.completion);
                    }

                    // Parse raw JSON to detect and filter server-side tool events,
                    // and handle client tool_use flow when tools are present
                    if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&event.data) {
                        let event_type = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let index = raw.get("index").and_then(|v| v.as_u64()).map(|v| v as usize);

                        match event_type {
                            "content_block_start" => {
                                if let Some(cb) = raw.get("content_block") {
                                    let block_type = cb.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                    match block_type {
                                        "tool_use" => {
                                            let name = cb.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                            let id = cb.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                            // Server-side tools have extra fields like "message", "integration_name"
                                            // or have builtin_ prefix
                                            let is_server_tool = name.starts_with("builtin_")
                                                || cb.get("message").and_then(|v| v.as_str()).is_some()
                                                || cb.get("integration_name").and_then(|v| v.as_str()).is_some();

                                            if is_server_tool {
                                                if let Some(idx) = index {
                                                    suppressed_indices.insert(idx);
                                                }
                                                debug!("[TOOL] suppressing server-side tool: {name}");
                                                continue;
                                            } else if has_tools {
                                                // Client tool - handle tool_use flow
                                                tool_use_id = Some(id.to_string());
                                                tool_use_index = index;
                                                debug!("[TOOL] detected client tool_use start: {id}");
                                            }
                                        }
                                        "tool_result" => {
                                            // Always suppress tool_result blocks (server-side only)
                                            if let Some(idx) = index {
                                                suppressed_indices.insert(idx);
                                            }
                                            debug!("[TOOL] suppressing tool_result block");
                                            continue;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_delta" | "content_block_stop" => {
                                if let Some(idx) = index {
                                    if suppressed_indices.contains(&idx) {
                                        continue;
                                    }
                                }
                                // Handle client tool_use stop
                                if has_tools && event_type == "content_block_stop" {
                                    if tool_use_index == index && tool_use_id.is_some() {
                                        debug!("[TOOL] client tool_use block ended, injecting stop events");
                                        // Yield the content_block_stop event first
                                        let e = SseEvent::default().event(event.event.clone()).id(event.id.clone());
                                        let e = if let Some(retry) = event.retry { e.retry(retry) } else { e };
                                        yield e.data(event.data.clone());

                                        // Inject message_delta with stop_reason=tool_use
                                        let delta_data = serde_json::json!({
                                            "type": "message_delta",
                                            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                                            "usage": {"output_tokens": 0}
                                        });
                                        yield SseEvent::default()
                                            .event("message_delta")
                                            .data(delta_data.to_string());

                                        // Inject message_stop
                                        let stop_data = serde_json::json!({"type": "message_stop"});
                                        yield SseEvent::default()
                                            .event("message_stop")
                                            .data(stop_data.to_string());

                                        // Register tool call for later tool_result
                                        if let Some(id) = tool_use_id.take() {
                                            TOOL_CALL_MANAGER.register(id, tool_call_session.clone()).await;
                                        }
                                        tool_call_break = true;
                                        break;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // Filter out Claude.ai-specific non-standard events
                    if event.event == "message_limit" {
                        continue;
                    }

                    let e = SseEvent::default().event(event.event).id(event.id);
                    let e = if let Some(retry) = event.retry { e.retry(retry) } else { e };
                    yield e.data(event.data);
                }
                // on end of stream, compute output tokens and persist totals (skip if tool_call)
                if !tool_call_break {
                    if !acc.is_empty() {
                        let mut out = None;
                        if enable_precise
                            && let Some(model) = last_params.as_ref().map(|p| p.model.clone())
                        {
                            out = count_code_output_tokens_for_text(
                                cookie.clone(), endpoint.clone(), proxy.clone(), client.clone(),
                                model, acc.clone(), handle.clone()
                            ).await.map(|v| v as u64);
                        }
                        let out = out.unwrap_or_else(|| {
                            let usage = crate::types::claude::Usage { input_tokens: input_tokens as u32, output_tokens: 0 };
                            let resp = crate::types::claude::CreateMessageResponse::text(acc.clone(), Default::default(), usage);
                            resp.count_tokens() as u64
                        });
                        if let Some(mut c) = cookie.clone() {
                            let family = last_params
                                .as_ref()
                                .map(|p| p.model.as_str())
                                .map(|m| {
                                    let m = m.to_ascii_lowercase();
                                    if m.contains("opus") {
                                        crate::config::ModelFamily::Opus
                                    } else if m.contains("sonnet") {
                                        crate::config::ModelFamily::Sonnet
                                    } else {
                                        crate::config::ModelFamily::Other
                                    }
                                })
                                .unwrap_or(crate::config::ModelFamily::Other);
                            c.add_and_bucket_usage(input_tokens, out, family);
                            let _ = handle.return_cookie(c, None).await;
                        }
                    } else if let Some(mut c) = cookie.clone() {
                        let family = last_params
                            .as_ref()
                            .map(|p| p.model.as_str())
                            .map(|m| {
                                let m = m.to_ascii_lowercase();
                                if m.contains("opus") {
                                    crate::config::ModelFamily::Opus
                                } else if m.contains("sonnet") {
                                    crate::config::ModelFamily::Sonnet
                                } else {
                                    crate::config::ModelFamily::Other
                                }
                            })
                            .unwrap_or(crate::config::ModelFamily::Other);
                        c.add_and_bucket_usage(input_tokens, 0, family);
                        let _ = handle.return_cookie(c, None).await;
                    }
                }
            };
            // normalize error type for axum SSE
            let stream = stream.map_err(|e: axum::Error| -> BoxError { e.into() });
            return Ok(Sse::new(stream)
                .keep_alive(Default::default())
                .into_response());
        }

        let stream = wreq_res.bytes_stream();
        let stream = stream.eventsource();
        let text = merge_sse(stream).await?;
        print_out_text(text.to_owned(), "claude_web_non_stream.txt");
        let mut response =
            CreateMessageResponse::text(text.clone(), Default::default(), self.usage.to_owned());

        // Prefer official counting if enabled
        let enable_precise = crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens;
        let mut usage = self.usage.to_owned();
        if enable_precise && let Some(inp) = self.try_code_count_tokens().await {
            usage.input_tokens = inp;
        }
        let mut output_tokens = response.count_tokens();
        if enable_precise && let Some(model) = self.last_params.as_ref().map(|p| p.model.clone()) {
            let out = count_code_output_tokens_for_text(
                self.cookie.clone(),
                self.endpoint.clone(),
                self.proxy.clone(),
                self.client.clone(),
                model,
                text.clone(),
                self.cookie_actor_handle.clone(),
            )
            .await;
            if let Some(v) = out {
                output_tokens = v;
            }
        }
        usage.output_tokens = output_tokens;
        response.usage = Some(usage.clone());
        self.persist_usage_totals(usage.input_tokens as u64, output_tokens as u64)
            .await;
        Ok(Json(response).into_response())
    }
}

async fn bearer_count_tokens(
    state: &ClaudeCodeState,
    access_token: &str,
    body: &CreateMessageParams,
) -> Option<u32> {
    let url = state.endpoint.join("v1/messages/count_tokens").ok()?;
    let resp = state
        .client
        .post(url.to_string())
        .bearer_auth(access_token)
        .header("anthropic-version", "2023-06-01")
        .json(body)
        .send()
        .await
        .ok()?;
    let resp = resp.check_claude().await.ok()?;
    let v: CountMessageTokensResponse = resp.json().await.ok()?;
    Some(v.input_tokens)
}

impl ClaudeWebState {
    pub(crate) async fn try_code_count_tokens(&mut self) -> Option<u32> {
        self.cookie.as_ref()?;
        let params = self.last_params.as_ref()?.clone();
        let mut code = ClaudeCodeState::new(self.cookie_actor_handle.clone());
        code.cookie = self.cookie.clone();
        code.endpoint = self.endpoint.clone();
        code.proxy = self.proxy.clone();
        code.client = self.client.clone();
        // populate cookie header for Claude code API requests
        if let Some(ref c) = self.cookie
            && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
        {
            code.set_cookie_header_value(val);
        }

        // OAuth exchange to get access token
        let org = code.get_organization().await.ok()?;
        let exch = code.exchange_code(&org).await.ok()?;
        code.exchange_token(exch).await.ok()?;
        let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

        // prepare body
        let mut body = params.clone();
        body.stream = Some(false);

        // do count_tokens
        bearer_count_tokens(&code, &access, &body).await
    }
}

async fn count_code_output_tokens_for_text(
    cookie: Option<crate::config::CookieStatus>,
    endpoint: Url,
    proxy: Option<Proxy>,
    client: wreq::Client,
    model: String,
    text: String,
    handle: crate::services::cookie_actor::CookieActorHandle,
) -> Option<u32> {
    let mut code = ClaudeCodeState::new(handle.clone());
    code.cookie = cookie.clone();
    code.endpoint = endpoint;
    code.proxy = proxy;
    code.client = client;
    if let Some(ref c) = cookie
        && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
    {
        code.set_cookie_header_value(val);
    }
    let org = code.get_organization().await.ok()?;
    let exch = code.exchange_code(&org).await.ok()?;
    code.exchange_token(exch).await.ok()?;
    let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

    let body = CreateMessageParams {
        model,
        messages: vec![Message::new_text(Role::Assistant, text)],
        ..Default::default()
    };
    // do not set count_tokens_allowed flag here to avoid races; handled by try_code_count_tokens
    bearer_count_tokens(&code, &access, &body).await
}
