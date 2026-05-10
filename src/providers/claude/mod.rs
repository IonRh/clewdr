use std::{sync::Arc, time::Instant};

use axum::response::Response;
use colored::Colorize;
use tracing::info;

use super::LLMProvider;
use crate::{
    claude_code_state::ClaudeCodeState,
    claude_web_state::ClaudeWebState,
    error::ClewdrError,
    middleware::claude::{ClaudeApiFormat, ClaudeContext},
    services::{cookie_actor::CookieActorHandle, tool_call::TOOL_CALL_MANAGER},
    types::claude::{ContentBlock, CreateMessageParams, MessageContent, Role},
    utils::{enabled, print_out_json},
};

#[derive(Clone, Copy)]
pub enum ClaudeOperation {
    Messages,
    CountTokens,
}

#[derive(Clone)]
pub struct ClaudeInvocation {
    pub params: CreateMessageParams,
    pub context: ClaudeContext,
    pub operation: ClaudeOperation,
}

impl ClaudeInvocation {
    pub fn messages(params: CreateMessageParams, context: ClaudeContext) -> Self {
        Self {
            params,
            context,
            operation: ClaudeOperation::Messages,
        }
    }

    pub fn count_tokens(params: CreateMessageParams, context: ClaudeContext) -> Self {
        Self {
            params,
            context,
            operation: ClaudeOperation::CountTokens,
        }
    }
}

pub struct ClaudeProviderResponse {
    pub context: ClaudeContext,
    pub response: Response,
}

struct ClaudeSharedState {
    cookie_actor_handle: CookieActorHandle,
}

impl ClaudeSharedState {
    fn new(cookie_actor_handle: CookieActorHandle) -> Self {
        Self {
            cookie_actor_handle,
        }
    }
}

#[derive(Clone)]
pub struct ClaudeProviders {
    web: Arc<ClaudeWebProvider>,
    code: Arc<ClaudeCodeProvider>,
}

impl ClaudeProviders {
    pub fn new(cookie_actor_handle: CookieActorHandle) -> Self {
        let shared = Arc::new(ClaudeSharedState::new(cookie_actor_handle));
        let web = Arc::new(ClaudeWebProvider::new(shared.clone()));
        let code = Arc::new(ClaudeCodeProvider::new(shared.clone()));
        Self { web, code }
    }

    pub fn web(&self) -> Arc<ClaudeWebProvider> {
        self.web.clone()
    }

    pub fn code(&self) -> Arc<ClaudeCodeProvider> {
        self.code.clone()
    }
}

#[derive(Clone)]
pub struct ClaudeWebProvider {
    shared: Arc<ClaudeSharedState>,
}

impl ClaudeWebProvider {
    fn new(shared: Arc<ClaudeSharedState>) -> Self {
        Self { shared }
    }
}

#[async_trait::async_trait]
impl LLMProvider for ClaudeWebProvider {
    type Request = ClaudeInvocation;
    type Output = ClaudeProviderResponse;

    async fn invoke(&self, request: Self::Request) -> Result<Self::Output, ClewdrError> {
        let mut state = ClaudeWebState::new(self.shared.cookie_actor_handle.clone());
        let stream = request.context.is_stream();
        state.api_format = request.context.api_format();
        state.stream = stream;
        state.usage = request.context.usage().to_owned();
        let ClaudeInvocation {
            params,
            context,
            operation,
        } = request;
        if !matches!(operation, ClaudeOperation::Messages) {
            return Err(ClewdrError::BadRequest {
                msg: "Unsupported operation for Claude Web",
            });
        }

        // Check if this is a tool_result request
        if let Some(tool_use_id) = extract_tool_result_id(&params) {
            info!("[TOOL] detected tool_result for: {tool_use_id}");
            if let Some(tool_state) = TOOL_CALL_MANAGER.take(&tool_use_id).await {
                let mut session = tool_state.session;
                session.stream = stream;
                session.api_format = context.api_format();
                session.usage = context.usage().to_owned();
                session.last_params = Some(params.clone());

                // Build tool_result payload from the last user message
                let tool_result_payload = build_tool_result_payload(&params);

                let stopwatch = Instant::now();
                let wreq_res = session.send_tool_result(tool_result_payload).await?;
                let response = session.transform_response(wreq_res).await?;
                let elapsed = stopwatch.elapsed();
                info!(
                    "[TOOL] tool_result response elapsed: {}s",
                    format!("{}", elapsed.as_secs_f32()).green()
                );
                return Ok(ClaudeProviderResponse { context, response });
            }
            info!("[TOOL] no pending session found, treating as normal request");
        }

        let format_display = match context.api_format() {
            ClaudeApiFormat::Claude => ClaudeApiFormat::Claude.to_string().green(),
            ClaudeApiFormat::OpenAI => ClaudeApiFormat::OpenAI.to_string().yellow(),
        };
        info!(
            "[REQ] stream: {}, msgs: {}, model: {}, think: {}, format: {}",
            enabled(stream),
            params.messages.len().to_string().green(),
            params.model.green(),
            enabled(params.thinking.is_some()),
            format_display
        );
        print_out_json(&params, "claude_web_client_req.json");
        let stopwatch = Instant::now();
        let response = state.try_chat(params).await?;
        let elapsed = stopwatch.elapsed();
        info!(
            "[FIN] elapsed: {}s",
            format!("{}", elapsed.as_secs_f32()).green()
        );
        Ok(ClaudeProviderResponse { context, response })
    }
}

/// Extract tool_use_id from the last user message if it contains a tool_result
fn extract_tool_result_id(params: &CreateMessageParams) -> Option<String> {
    let last_msg = params.messages.last()?;
    if last_msg.role != Role::User {
        return None;
    }
    match &last_msg.content {
        MessageContent::Blocks { content } => {
            // Find the last tool_result block
            content.iter().rev().find_map(|block| {
                if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                    Some(tool_use_id.clone())
                } else {
                    None
                }
            })
        }
        _ => None,
    }
}

/// Build the tool_result payload to send to Claude.ai's /tool_result endpoint
fn build_tool_result_payload(params: &CreateMessageParams) -> serde_json::Value {
    let last_msg = params.messages.last().unwrap();
    match &last_msg.content {
        MessageContent::Blocks { content } => {
            // Find the last tool_result block and serialize it
            let tool_result = content.iter().rev().find_map(|block| {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    ..
                } = block
                {
                    Some(serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                        "is_error": is_error.unwrap_or(false),
                    }))
                } else {
                    None
                }
            });
            tool_result.unwrap_or(serde_json::json!({}))
        }
        _ => serde_json::json!({}),
    }
}

#[derive(Clone)]
pub struct ClaudeCodeProvider {
    shared: Arc<ClaudeSharedState>,
}

impl ClaudeCodeProvider {
    fn new(shared: Arc<ClaudeSharedState>) -> Self {
        Self { shared }
    }
}

#[async_trait::async_trait]
impl LLMProvider for ClaudeCodeProvider {
    type Request = ClaudeInvocation;
    type Output = ClaudeProviderResponse;

    async fn invoke(&self, request: Self::Request) -> Result<Self::Output, ClewdrError> {
        let mut state = ClaudeCodeState::new(self.shared.cookie_actor_handle.clone());
        state.api_format = request.context.api_format();
        state.stream = request.context.is_stream();
        state.system_prompt_hash = request.context.system_prompt_hash();
        state.anthropic_beta_header = request.context.anthropic_beta().map(str::to_string);
        state.usage = request.context.usage().to_owned();
        let ClaudeInvocation {
            params,
            context,
            operation,
        } = request;
        match operation {
            ClaudeOperation::Messages => {
                let format_display = match context.api_format() {
                    ClaudeApiFormat::Claude => ClaudeApiFormat::Claude.to_string().green(),
                    ClaudeApiFormat::OpenAI => ClaudeApiFormat::OpenAI.to_string().yellow(),
                };
                info!(
                    "[REQ] stream: {}, msgs: {}, model: {}, format: {}",
                    enabled(state.stream),
                    params.messages.len().to_string().green(),
                    params.model.green(),
                    format_display
                );
                print_out_json(&params, "claude_code_client_req.json");
                let stopwatch = Instant::now();
                let response = state.try_chat(params).await?;
                let elapsed = stopwatch.elapsed();
                info!(
                    "[FIN] elapsed: {}s",
                    format!("{}", elapsed.as_secs_f32()).green()
                );
                Ok(ClaudeProviderResponse { context, response })
            }
            ClaudeOperation::CountTokens => {
                info!(
                    "[TOKENS] msgs: {}, model: {}",
                    params.messages.len().to_string().green(),
                    params.model.green()
                );
                let stopwatch = Instant::now();
                let response = state.try_count_tokens(params, context.is_web()).await?;
                let elapsed = stopwatch.elapsed();
                info!(
                    "[TOKENS] elapsed: {}s",
                    format!("{}", elapsed.as_secs_f32()).green()
                );
                Ok(ClaudeProviderResponse { context, response })
            }
        }
    }
}

pub fn build_providers(cookie_actor_handle: CookieActorHandle) -> ClaudeProviders {
    ClaudeProviders::new(cookie_actor_handle)
}
