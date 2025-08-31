use std::fmt::{Debug, Write};

use tracing::{field::Visit, level_filters::LevelFilter, span::Id, Event, Level, Subscriber};
use tracing_subscriber::{field::{MakeExt, RecordFields}, filter, fmt::{self, format::{self, Writer}, FmtContext, FormatEvent, FormatFields, FormattedFields}, layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt, Layer};
use colored::*;
use anyhow::Error;

#[derive(PartialEq, PartialOrd, Ord, Eq, Clone, Debug)]
struct FieldData {
    tag: Option<String>
}

impl Default for FieldData {
    fn default() -> Self {
        Self {
            tag: None
        }
    }
}

#[derive(Default)]
struct TagExtractor {
    data: FieldData
}

impl Visit for TagExtractor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "tag" {
            self.data.tag = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
}

#[derive(Default)]
struct MessageExtractor {
    message: String
}

impl Visit for MessageExtractor {
    fn record_debug(&mut self, field: &tracing::field::Field, debug: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", debug);
        }
    }
}

struct CustomLayer;

impl<S> Layer<S> for CustomLayer
where
    S: tracing::Subscriber,
    S: for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = TagExtractor::default();
        attrs.values().record(&mut visitor);

        let span = ctx.span(id).expect("context does not include new span");
        let mut extensions = span.extensions_mut();
        extensions.insert::<FieldData>(visitor.data);
    }
}

fn color_level(level: Level) -> ColoredString {
    match level {
        Level::ERROR => "error".bright_red().bold(),
        Level::WARN => "warning".yellow().bold(),
        Level::INFO => "info".blue(),
        Level::DEBUG => "debug".bright_yellow(),
        Level::TRACE => "trc".bright_white() // "trc" not "trace" because white + white less
                                             // visible so distinction is provided by a shorter
                                             // length string
    }
}

fn color_tag(tag: String) -> ColoredString {
    match tag.as_str() {
        "main" => tag.red(),
        "gpu" => tag.cyan(),
        "assets" => tag.bright_blue(),
        "audio" => tag.bright_green(),
        _ => tag.bright_black()
    }
}

struct TaggedFormatter;

impl<S, N> FormatEvent<S, N> for TaggedFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let time = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let metadata = event.metadata();
        let level = *metadata.level();


        let data: FieldData = if let Some(leaf_span) = ctx.parent_span() {
            let mut data = None;
            for span in leaf_span.scope().from_root() {
                let ext = span.extensions();
                let field_data = ext.get::<FieldData>().expect("no fielddata");
                if field_data != &FieldData::default() {
                    data = Some(field_data.clone());
                }
            }

            data
        } else {
            None
        }.unwrap_or(FieldData::default());

        let tag = color_tag(data.tag.unwrap_or(String::from("other")));

        let mut visitor = MessageExtractor::default();
        event.record(&mut visitor);

        if event.fields().find(|f| f.name() == "help").is_some(){
            write!(writer, "{:<6}{}  ", tag, color_level(level))?;
            write!(writer, "{}", format!("help: {}", visitor.message).yellow())?;
        } else if level == Level::ERROR {
            write!(writer, "{:<6}{}  ", tag, color_level(level))?;
            write!(writer, "{}", visitor.message)?;
        } else {
            write!(writer, "{} {:<6}{:>8}  ", time.bright_black(), tag, color_level(level))?;
            write!(writer, "{}", visitor.message)?;
        }

        writeln!(writer)
    }
}

#[derive(clap::ValueEnum, Clone, Default, Debug)]
pub enum Verbosity {
    ProblemsOnly,
    #[default]
    Normal,
    Debug,
    Everything,
}

// there are easier ways, but i feel like such a rustacean today, so traits it is
impl Into<Level> for Verbosity {
    fn into(self) -> Level {
        match self {
            Verbosity::Normal => Level::INFO,
            Verbosity::ProblemsOnly => Level::WARN,
            Verbosity::Debug => Level::DEBUG,
            Verbosity::Everything => Level::TRACE,
        }
    }
}

pub fn setup<I: Into<Level>>(max_level: I) -> Result<(), Error> {
    let max_level: Level = max_level.into();
    let enable_log = max_level >= Level::TRACE;
    tracing_subscriber::registry()
        .with(CustomLayer)
        .with(LevelFilter::from_level(max_level))
        .with(
            fmt::layer()
                .event_format(TaggedFormatter)
                //.map_fmt_fields(|f| f.debug_alt())
                .with_filter(filter::filter_fn(move |metadata| { 
                    let from_current = metadata.target().starts_with(env!("CARGO_CRATE_NAME"));
                    from_current || (!from_current && enable_log)
                }))
        )
        .init();
    
    Ok(())
}
