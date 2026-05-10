use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::claude_web_state::ClaudeWebState;

/// Global tool call manager instance
pub static TOOL_CALL_MANAGER: LazyLock<ToolCallManager> = LazyLock::new(ToolCallManager::new);

/// State for a pending tool call - holds the session needed to send tool_result
#[derive(Clone)]
pub struct ToolCallState {
    pub session: ClaudeWebState,
    pub created_at: Instant,
}

/// Manages pending tool calls, mapping tool_use_id to session state
#[derive(Clone)]
pub struct ToolCallManager {
    calls: Arc<Mutex<HashMap<String, ToolCallState>>>,
}

impl ToolCallManager {
    pub fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a pending tool call
    pub async fn register(&self, tool_use_id: String, session: ClaudeWebState) {
        info!("[TOOL] registered tool_call: {tool_use_id}");
        self.calls.lock().await.insert(
            tool_use_id,
            ToolCallState {
                session,
                created_at: Instant::now(),
            },
        );
    }

    /// Take a pending tool call (removes it from the map)
    pub async fn take(&self, tool_use_id: &str) -> Option<ToolCallState> {
        let state = self.calls.lock().await.remove(tool_use_id);
        if state.is_some() {
            info!("[TOOL] resolved tool_call: {tool_use_id}");
        }
        state
    }

    /// Cleanup expired tool calls (older than timeout)
    #[allow(dead_code)]
    pub async fn cleanup(&self, timeout: Duration) {
        let mut calls = self.calls.lock().await;
        let before = calls.len();
        calls.retain(|id, state| {
            let expired = state.created_at.elapsed() > timeout;
            if expired {
                warn!("[TOOL] expired tool_call: {id}");
            }
            !expired
        });
        let removed = before - calls.len();
        if removed > 0 {
            info!("[TOOL] cleaned up {removed} expired tool calls");
        }
    }
}
