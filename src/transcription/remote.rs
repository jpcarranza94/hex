//! A transcription server of the user's own: the finished 16 kHz clip goes
//! as a WAV to the OpenAI-compatible `/v1/audio/transcriptions` URL in
//! `settings.json` (`remote_transcription`), and the reply's `text` comes
//! back. Nothing loads on this computer, nothing streams, and there are no
//! word timings. Speech model "Remote server" is offered only while a URL
//! is set, and only on macOS, where `Transcriber` can run it (#603).
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::sync::RwLock;
use std::time::Duration;

use color_eyre::eyre::{Result, WrapErr, eyre};
use serde::{Deserialize, Serialize};

/// `settings.json`'s `remote_transcription`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct RemoteTranscription {
    /// The whole endpoint, for example
    /// `https://server.example.ts.net/v1/audio/transcriptions`.
    pub url: String,
    /// The server's model name, sent as the `model` field when set.
    pub model: String,
    /// Sent as `Authorization: Bearer`, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

static CONFIGURED: RwLock<Option<RemoteTranscription>> = RwLock::new(None);

/// Publishes the saved server, as settings apply; one without a URL is none.
pub fn set_configured(server: Option<&RemoteTranscription>) {
    *CONFIGURED
        .write()
        .unwrap_or_else(|error| error.into_inner()) = server
        .filter(|server| !server.url.trim().is_empty())
        .cloned();
}

/// The server clips go to, when one is set.
pub fn configured() -> Option<RemoteTranscription> {
    CONFIGURED
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
}

/// Upload and transcription of one clip together.
// ponytail: one fixed deadline; scale it with the clip's length if long dictations time out.
const DEADLINE: Duration = Duration::from_secs(30);

const MAX_REPLY_BYTES: u64 = 1024 * 1024;

const BOUNDARY: &str = "hex-remote-transcription-7f3a9c";

/// The runtime behind "Remote server". It holds nothing: each clip reads the
/// server from settings, so a changed URL takes effect on the next one.
#[cfg(target_os = "macos")]
pub struct Remote;

#[cfg(target_os = "macos")]
impl Remote {
    pub fn load() -> Result<Self> {
        configured().map(|_| Self).ok_or_else(not_configured)
    }

    /// Transcribes normalized 16 kHz audio on the server.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let server = configured().ok_or_else(not_configured)?;
        let host = url::Url::parse(&server.url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_owned))
            .unwrap_or_else(|| server.url.clone());
        let (content_type, body) = multipart(&server.model, &wav(samples)?);
        let mut request = ureq::http::Request::builder()
            .method("POST")
            .uri(server.url.as_str())
            .header("Content-Type", content_type);
        if let Some(key) = server.api_key.as_deref().filter(|key| !key.is_empty()) {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let request = request
            .body(body)
            .map_err(|error| eyre!("remote_transcription.url is not a usable URL: {error}"))?;
        let agent = crate::net::agent(&server.url, |config| config.timeout_global(Some(DEADLINE)));
        let reach =
            |error: ureq::Error| eyre!("Couldn't reach {host}: {}", crate::net::reason(&error));
        let reply = agent
            .run(request)
            .map_err(reach)?
            .body_mut()
            .with_config()
            .limit(MAX_REPLY_BYTES)
            .read_to_vec()
            .map_err(reach)?;
        text(&reply).wrap_err_with(|| format!("{host} answered without a transcript"))
    }
}

#[cfg(target_os = "macos")]
fn not_configured() -> color_eyre::Report {
    eyre!("Remote server has no URL: set remote_transcription.url in settings")
}

/// The clip as 16-bit 16 kHz mono WAV, which every server reads.
fn wav(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut bytes = std::io::Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    let mut writer = hound::WavWriter::new(&mut bytes, spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)?;
    }
    writer.finalize()?;
    Ok(bytes.into_inner())
}

/// The `multipart/form-data` content type and body the endpoint takes.
fn multipart(model: &str, wav: &[u8]) -> (String, Vec<u8>) {
    let mut body = Vec::with_capacity(wav.len() + 512);
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    };
    if !model.is_empty() {
        field("model", model);
    }
    field("response_format", "json");
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"dictation.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={BOUNDARY}"), body)
}

/// The transcript in a `response_format=json` reply.
fn text(reply: &[u8]) -> Result<String> {
    #[derive(Deserialize)]
    struct Reply {
        text: String,
    }
    let reply: Reply = serde_json::from_slice(reply)?;
    Ok(reply.text.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::{BOUNDARY, multipart, text, wav};

    #[test]
    fn the_request_carries_the_model_format_and_wav() {
        let audio = wav(&[0.0, 0.5, -0.5, 1.5]).unwrap();
        assert_eq!(&audio[..4], b"RIFF");
        assert_eq!(audio.len(), 44 + 4 * 2);

        let (content_type, body) = multipart("parakeet", &audio);
        assert_eq!(
            content_type,
            format!("multipart/form-data; boundary={BOUNDARY}")
        );
        let head = String::from_utf8_lossy(&body[..body.len() - audio.len()]);
        assert!(head.contains("name=\"model\"\r\n\r\nparakeet\r\n"));
        assert!(head.contains("name=\"response_format\"\r\n\r\njson\r\n"));
        assert!(head.contains("name=\"file\"; filename=\"dictation.wav\""));
        assert!(body.windows(audio.len()).any(|window| window == audio));
        assert!(body.ends_with(format!("\r\n--{BOUNDARY}--\r\n").as_bytes()));

        let (_, without_model) = multipart("", &audio);
        assert!(!String::from_utf8_lossy(&without_model).contains("name=\"model\""));
    }

    #[test]
    fn the_reply_text_is_trimmed_and_other_fields_ignored() {
        assert_eq!(
            text(br#"{"text":" Hello there. ","sentences":[]}"#).unwrap(),
            "Hello there."
        );
        assert!(text(br#"{"error":"no"}"#).is_err());
    }
}
