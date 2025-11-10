use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::client_common::tools::ToolSpec;
use crate::codex::compact::content_items_to_text;
use crate::error::Result as CodexResult;
use crate::model_family::ModelFamily;
use crate::model_provider_info::ModelProviderInfo;
use crate::model_provider_info::WireApi;
use crate::protocol::TokenUsage;
use chrono::Utc;
use codex_protocol::ConversationId;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::task;
use tracing::warn;
use std::fmt;

const TRAJECTORY_PATH_ENV: &str = "CODEX_TRAJECTORY_PATH";

// 静态变量保存进程级别的时间戳，确保整个会话使用同一个文件
static PROCESS_TIMESTAMP: OnceLock<String> = OnceLock::new();

pub(crate) struct TrajectoryRecorder {
    path: PathBuf,
    provider: String,
    model: String,
    wire_api: String,
    state: Mutex<TrajectoryState>,
}

impl fmt::Debug for TrajectoryRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrajectoryRecorder")
            .field("path", &self.path)
            .field("provider", &self.provider)
            .field("model", &self.model)
            .finish()
    }
}

struct TrajectoryState {
    interaction: TrajectoryInteraction,
    response: InteractionResponseBuilder,
    error: Option<String>,
    written: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TrajectoryInteraction {
    timestamp: String,
    conversation_id: String,
    provider: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    wire_api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_task: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_messages: Vec<LoggedMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools_available: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<InteractionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LoggedMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct InteractionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    output_items: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<TokenUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

#[derive(Clone, Default)]
struct InteractionResponseBuilder {
    content: String,
    reasoning_summary: String,
    reasoning_content: String,
    output_items: Vec<Value>,
    response_id: Option<String>,
    usage: Option<TokenUsage>,
    finish_reason: Option<String>,
}

fn make_interaction(
    provider_name: &str,
    model: &str,
    wire_api_label: &str,
    model_family: &ModelFamily,
    conversation_id: &ConversationId,
    prompt: Option<&Prompt>,
) -> TrajectoryInteraction {
    let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string();

    let (input_messages, current_task, tools_available) = if let Some(p) = prompt {
        let instructions = p.get_full_instructions(model_family).to_string();
        let formatted_input = p.get_formatted_input();
        let (msgs, task) = build_logged_messages(instructions, &formatted_input);
        let tools = serialize_tools(&p.tools);
        (msgs, task, tools)
    } else {
        (Vec::new(), None, None)
    };

    TrajectoryInteraction {
        timestamp,
        conversation_id: conversation_id.to_string(),
        provider: provider_name.to_string(),
        model: model.to_string(),
        wire_api: Some(wire_api_label.to_string()),
        current_task,
        input_messages,
        tools_available,
        response: None,
        error: None,
    }
}

impl TrajectoryRecorder {
    pub(crate) fn new(
        provider: &ModelProviderInfo,
        model: &str,
        model_family: &ModelFamily,
        conversation_id: &ConversationId,
        target_path: Option<PathBuf>,
        prompt: Option<&Prompt>,
    ) -> Option<Arc<Self>> {
        let Some(path) = resolve_target_path(target_path) else { return None; };
        if let Err(e) = init_trajectory_file(&path) {
            warn!("Failed to init trajectory file {}: {e:?}", path.display());
            return None;
        }

        let wire_api_label = wire_api_label(provider.wire_api).to_string();
        let interaction = make_interaction(
            &provider.name,
            model,
            &wire_api_label,
            model_family,
            conversation_id,
            prompt,
        );

        Some(Arc::new(Self {
            path,
            provider: provider.name.clone(),
            model: model.to_string(),
            wire_api: wire_api_label, // <--- 保存
            state: Mutex::new(TrajectoryState {
                interaction,
                response: InteractionResponseBuilder::default(),
                error: None,
                written: false,
            }),
        }))
    }

    pub(crate) fn wrap(
        recorder: Arc<Self>,
        mut rx_event: mpsc::Receiver<CodexResult<ResponseEvent>>,
    ) -> ResponseStream {
        let (tx, rx_logged) = mpsc::channel(1600);
        task::spawn(async move {
            while let Some(event) = rx_event.recv().await {
                recorder.record_event(&event).await;
                if tx.send(event).await.is_err() {
                    break;
                }
            }
            recorder.finish().await;
        });
        ResponseStream {
            rx_event: rx_logged,
        }
    }

    pub(crate) async fn record_event(&self, event: &CodexResult<ResponseEvent>) {
        let mut state = self.state.lock().await;
        match event {
            Ok(ResponseEvent::OutputTextDelta(delta)) => {
                state.response.push_text(delta);
            }
            Ok(ResponseEvent::ReasoningSummaryDelta(delta)) => {
                state.response.push_reasoning_summary(delta);
            }
            Ok(ResponseEvent::ReasoningSummaryPartAdded) => {
                state.response.push_reasoning_summary("\n");
            }
            Ok(ResponseEvent::ReasoningContentDelta(delta)) => {
                state.response.push_reasoning_content(delta);
            }
            Ok(ResponseEvent::OutputItemDone(item)) => match serde_json::to_value(item) {
                Ok(value) => state.response.add_output_item(value),
                Err(err) => warn!("Failed to serialize ResponseItem for trajectory log: {err:?}"),
            },
            Ok(ResponseEvent::Completed {
                response_id,
                token_usage,
            }) => {
                state
                    .response
                    .set_completed(response_id, token_usage.as_ref());
            }
            Err(err) => {
                if state.error.is_none() {
                    state.error = Some(err.to_string());
                }
            }
            _ => {}
        }
    }

    pub(crate) async fn finish(&self) {
        let (interaction, path, provider, model) = {
            let mut state = self.state.lock().await;
            if state.written {
                return;
            }
            state.written = true;

            let mut interaction = state.interaction.clone();
            let builder = std::mem::take(&mut state.response);
            interaction.response = builder.build(&self.model);
            if let Some(error) = state.error.clone() {
                interaction.error = Some(error);
            }
            (
                interaction,
                self.path.clone(),
                self.provider.clone(),
                self.model.clone(),
            )
        };

        let handle = task::spawn_blocking(move || {
            if let Err(err) = persist_interaction(&path, &provider, &model, &interaction) {
                warn!(
                    "Failed to persist trajectory interaction to {}: {err:?}",
                    path.display()
                );
            }
        });

        if let Err(err) = handle.await {
            warn!("Trajectory logging task failed: {err:?}");
        }
    }

    pub(crate) async fn start_new_interaction(
        &self,
        model_family: &ModelFamily,
        conversation_id: &ConversationId,
        prompt: Option<&Prompt>,
    ) {
        {
            let state = self.state.lock().await;
            if !state.written {
            } else {
            }
        }
        let need_finish = {
            let state = self.state.lock().await;
            !state.written
        };
        if need_finish {
            self.finish().await;
        }

        let mut state = self.state.lock().await;
        let interaction = make_interaction(
            &self.provider,
            &self.model,
            &self.wire_api,
            model_family,
            conversation_id,
            prompt,
        );
        *state = TrajectoryState {
            interaction,
            response: InteractionResponseBuilder::default(),
            error: None,
            written: false,
        };
    }
}

impl InteractionResponseBuilder {
    fn push_text(&mut self, delta: &str) {
        self.content.push_str(delta);
    }

    fn push_reasoning_summary(&mut self, delta: &str) {
        self.reasoning_summary.push_str(delta);
    }

    fn push_reasoning_content(&mut self, delta: &str) {
        self.reasoning_content.push_str(delta);
    }

    fn add_output_item(&mut self, value: Value) {
        self.output_items.push(value);
    }

    fn set_completed(&mut self, response_id: &str, usage: Option<&TokenUsage>) {
        self.response_id = Some(response_id.to_string());
        if let Some(usage) = usage {
            self.usage = Some(usage.clone());
        }
    }

    fn build(self, model: &str) -> Option<InteractionResponse> {
        let has_output = !self.content.is_empty()
            || !self.reasoning_summary.is_empty()
            || !self.reasoning_content.is_empty()
            || !self.output_items.is_empty()
            || self.response_id.is_some()
            || self.usage.is_some()
            || self.finish_reason.is_some();

        if !has_output {
            return None;
        }

        Some(InteractionResponse {
            content: (!self.content.is_empty()).then_some(self.content),
            reasoning_summary: (!self.reasoning_summary.is_empty())
                .then_some(self.reasoning_summary),
            reasoning_content: (!self.reasoning_content.is_empty())
                .then_some(self.reasoning_content),
            output_items: self.output_items,
            response_id: self.response_id,
            usage: self.usage,
            finish_reason: self.finish_reason,
            model: Some(model.to_string()),
        })
    }
}

fn resolve_target_path(preferred: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = preferred {
        return Some(p);
    }
    let timestamp = PROCESS_TIMESTAMP.get_or_init(|| {
        Utc::now().format("%Y%m%d_%H%M%S").to_string()
    });
    match std::env::var(TRAJECTORY_PATH_ENV) {
        Ok(raw_path) => {
            if raw_path.trim().is_empty() {
                return None;
            }

            let candidate = PathBuf::from(raw_path);
            let is_json_file = {
                match candidate.extension().and_then(|ext| ext.to_str()) {
                    Some(ext) => ext.eq_ignore_ascii_case("json"),
                    None => false,
                }
            };
            if is_json_file {
                if let Some(parent) = candidate.parent() {
                    if !parent.as_os_str().is_empty() {
                        if let Err(_e) = fs::create_dir_all(parent) {
                            return None;
                        }
                    }
                }
                Some(candidate)
            } else {
                if let Err(_e) = fs::create_dir_all(&candidate) {
                    return None;
                }
                let file_name = format!("codex_trajectory_{timestamp}.json");
                Some(candidate.join(file_name))
            }
        }
        Err(_) => {
            Some(PathBuf::from(format!("codex_trajectory_{timestamp}.json")))
        }
    }
}

fn init_trajectory_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    if !path.exists() {
        fs::write(path, "{}")?;
    }
    Ok(())
}

fn wire_api_label(wire_api: WireApi) -> &'static str {
    match wire_api {
        WireApi::Responses => "responses",
        WireApi::Chat => "chat",
    }
}

const MAX_LOG_CHARS: usize = 4000;

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let mut out = s[..max].to_string();
    out.push_str("…[truncated]");
    out
}

fn to_compact_json(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "<json-serialize-error>".to_string())
}

fn build_logged_messages(
    instructions: String,
    input: &[ResponseItem],
) -> (Vec<LoggedMessage>, Option<String>) {
    let mut messages = vec![LoggedMessage {
        role: "system".to_string(),
        content: instructions,
    }];

    let mut last_user: Option<String> = None;

    for item in input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                if let Some(text) = content_items_to_text(content) {
                    let txt = text.trim();
                    if !txt.is_empty() {
                        messages.push(LoggedMessage {
                            role: role.clone(),
                            content: truncate(txt, MAX_LOG_CHARS),
                        });
                        if role == "user" {
                            last_user = Some(txt.to_string());
                        }
                    }
                }
            }

            ResponseItem::Reasoning { summary, content, .. } => {
                if let Some(text) = reasoning_to_text(summary, content) {
                    let txt = text.trim();
                    if !txt.is_empty() {
                        messages.push(LoggedMessage {
                            role: "reasoning".to_string(),
                            content: truncate(txt, MAX_LOG_CHARS),
                        });
                    }
                }
            }

            ResponseItem::LocalShellCall { status, action, .. } => {
                let v = serde_json::json!({
                    "type": "local_shell_call",
                    "status": status,
                    "action": action,
                });
                messages.push(LoggedMessage {
                    role: "local_shell_call".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }

            ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
                let args_json: serde_json::Value = serde_json::from_str(arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(arguments.clone()));
                let v = serde_json::json!({
                    "type": "function_call",
                    "name": name,
                    "call_id": call_id,
                    "arguments": args_json,
                });
                messages.push(LoggedMessage {
                    role: "function_call".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }

            ResponseItem::FunctionCallOutput { call_id, output } => {
                let v = serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                });
                messages.push(LoggedMessage {
                    role: "function_call_output".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }

            ResponseItem::CustomToolCall { status, call_id, name, input, .. } => {
                let parsed_input: serde_json::Value = serde_json::from_str(input)
                    .unwrap_or_else(|_| serde_json::Value::String(input.clone()));
                let v = serde_json::json!({
                    "type": "custom_tool_call",
                    "status": status,
                    "call_id": call_id,
                    "name": name,
                    "input": parsed_input,
                });
                messages.push(LoggedMessage {
                    role: "custom_tool_call".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }

            ResponseItem::CustomToolCallOutput { call_id, output } => {
                let parsed_output: serde_json::Value = serde_json::from_str(output)
                    .unwrap_or_else(|_| serde_json::Value::String(output.clone()));
                let v = serde_json::json!({
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": parsed_output,
                });
                messages.push(LoggedMessage {
                    role: "custom_tool_call_output".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }

            ResponseItem::WebSearchCall { status, action, .. } => {
                let v = serde_json::json!({
                    "type": "web_search_call",
                    "status": status,
                    "action": action,
                });
                messages.push(LoggedMessage {
                    role: "web_search_call".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }

            ResponseItem::Other => {
                let v = serde_json::json!({
                    "type": "other_input_item"
                });
                messages.push(LoggedMessage {
                    role: "tool_output".to_string(),
                    content: truncate(&to_compact_json(&v), MAX_LOG_CHARS),
                });
            }
        }
    }

    (messages, last_user)
}


fn reasoning_to_text(
    summary: &[ReasoningItemReasoningSummary],
    content: &Option<Vec<ReasoningItemContent>>,
) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    for ReasoningItemReasoningSummary::SummaryText { text } in summary {
        if !text.trim().is_empty() {
            parts.push(text.clone());
        }
    }

    if let Some(items) = content {
        for item in items {
            match item {
                ReasoningItemContent::ReasoningText { text }
                | ReasoningItemContent::Text { text } => {
                    if !text.trim().is_empty() {
                        parts.push(text.clone());
                    }
                }
            }
        }
    }

    (!parts.is_empty()).then_some(parts.join("\n"))
}

fn serialize_tools(tools: &[ToolSpec]) -> Option<Vec<String>> {
    if tools.is_empty() {
        return None;
    }

    let names: Vec<String> = tools.iter().map(|tool| tool.name().to_string()).collect();
    (!names.is_empty()).then_some(names)
}

fn persist_interaction(
    path: &Path,
    provider: &str,
    model: &str,
    interaction: &TrajectoryInteraction,
) -> anyhow::Result<()> {
    let existing = fs::read_to_string(path).ok();
    let mut doc: Value = existing
        .as_deref()
        .and_then(|content| serde_json::from_str(content).ok())
        .unwrap_or_else(|| json!({}));

    if !doc.is_object() {
        doc = json!({});
    }

    let obj = doc.as_object_mut().expect("document checked to be object");

    append_interaction(obj, interaction)?;
    populate_metadata(obj, provider, model, interaction);

    let content = serde_json::to_string_pretty(&doc)?;
    fs::write(path, content)?;
    Ok(())
}

fn append_interaction(
    obj: &mut Map<String, Value>,
    interaction: &TrajectoryInteraction,
) -> anyhow::Result<()> {
    let entry = obj
        .entry("llm_interactions".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));

    let interaction_value = serde_json::to_value(interaction)?;
    if let Value::Array(items) = entry {
        items.push(interaction_value);
    } else {
        *entry = Value::Array(vec![interaction_value]);
    }
    Ok(())
}

fn populate_metadata(
    obj: &mut Map<String, Value>,
    provider: &str,
    model: &str,
    interaction: &TrajectoryInteraction,
) {
    let start_ts = interaction.timestamp.clone();
    set_string_if_empty(obj, "start_time", start_ts);
    let end_ts = Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string();
    obj.insert("end_time".to_string(), Value::String(end_ts));
    obj.insert("provider".to_string(), Value::String(provider.to_string()));
    obj.insert("model".to_string(), Value::String(model.to_string()));
    obj.insert(
        "conversation_id".to_string(),
        Value::String(interaction.conversation_id.clone()),
    );
    obj.insert(
        "success".to_string(),
        Value::Bool(interaction.error.is_none()),
    );

    if let Some(task) = interaction.current_task.as_ref() {
        if !task.trim().is_empty() {
            obj.insert("task".to_string(), Value::String(task.clone()));
        }
    }
    if let Some(response) = interaction.response.as_ref() {
        if let Some(usage) = response.usage.as_ref() {
            add_to_counter(obj, "total_input_tokens", usage.input_tokens);
            add_to_counter(obj, "total_output_tokens", usage.output_tokens);
            add_to_counter(obj, "total_tokens", usage.total_tokens);
        }
    }
}

fn set_string_if_empty(map: &mut Map<String, Value>, key: &str, value: String) {
    let needs_update = match map.get(key).and_then(Value::as_str) {
        Some(s) => s.is_empty(),
        None => true,
    };
    if needs_update {
        map.insert(key.to_string(), Value::String(value));
    }
}

fn add_to_counter(map: &mut Map<String, Value>, key: &str, delta: u64) {
    let current = map.get(key).and_then(Value::as_u64).unwrap_or(0);
    let updated = current.saturating_add(delta);
    map.insert(key.to_string(), Value::Number(updated.into()));
}
