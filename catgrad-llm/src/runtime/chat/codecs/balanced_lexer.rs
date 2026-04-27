//! Balanced-bracket / string-aware top-level lexer.
//!
//! The Pythonic and Gemma 4 codecs both need to split a flat key-value
//! list on a separator while respecting:
//!
//! - balanced brackets (`{}`, `[]`, optionally `()`),
//! - paired strings (ASCII single/double quotes, or a paired multi-byte
//!   sentinel like Gemma 4's `<|"|>`).
//!
//! Each protocol's lexer used to be ~80 LOC of nearly-identical code;
//! this module factors the shared machinery out behind a small config.
//!
//! The lexer doesn't tokenise values — it only finds top-level
//! separator positions. The caller parses the spans between them.

/// How the wire format quotes string literals — drives whether the
/// lexer treats a stretch of bytes as opaque (quoted) or as
/// participating in bracket-balancing.
#[derive(Clone, Copy)]
pub enum StringQuote {
    /// ASCII `'` and `"` both work as string delimiters; matched by
    /// the same character (Python / JSON style).
    Ascii,
    /// A paired multi-byte sentinel — the same string opens and closes
    /// (Gemma 4's `<|"|>` is the in-tree case). Strings cannot be
    /// nested.
    PairedSentinel(&'static str),
}

#[derive(Clone, Copy)]
pub struct BalancedConfig {
    pub string_quote: StringQuote,
    /// Top-level separator. The lexer also walks `find_top_level` for
    /// any other character; this field is used by `split_top_level`.
    pub separator: char,
    /// Whether `(` `)` participate in bracket-balancing alongside
    /// `{}` `[]`. Pythonic includes them; JSON / Gemma 4 do not (they
    /// have no parenthesised expressions in arg payloads).
    pub include_parens: bool,
}

impl BalancedConfig {
    pub const PYTHONIC: Self = Self {
        string_quote: StringQuote::Ascii,
        separator: ',',
        include_parens: true,
    };
    pub const GEMMA4: Self = Self {
        string_quote: StringQuote::PairedSentinel("<|\"|>"),
        separator: ',',
        include_parens: false,
    };
}

/// Split `text` at top-level occurrences of `cfg.separator`. Empty
/// trimmed parts are dropped (mirrors both codecs' historical
/// behaviour). Returns `Err(unterminated)` on unbalanced brackets or
/// an unclosed string.
pub fn split_top_level<'a>(text: &'a str, cfg: &BalancedConfig) -> Result<Vec<&'a str>, String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut walker = Walker::new(text, cfg);
    while let Some(idx) = walker.advance_to_separator()? {
        let part = text[start..idx].trim();
        if !part.is_empty() {
            parts.push(part);
        }
        start = idx + cfg.separator.len_utf8();
        walker.skip(cfg.separator.len_utf8());
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    Ok(parts)
}

/// Find the byte index of the first top-level occurrence of `needle`.
/// Returns `Ok(None)` if the input is balanced and contains no such
/// character. `Err` on unbalanced brackets / unclosed string.
pub fn find_top_level<'a>(
    text: &'a str,
    needle: char,
    cfg: &BalancedConfig,
) -> Result<Option<usize>, String> {
    let mut walker = Walker::new(text, cfg);
    walker.find_char(needle)
}

/// Inner state machine. Walks `text` once, tracking bracket depth and
/// in-string state. Each public method consumes the rest of the text
/// looking for a target.
struct Walker<'a> {
    text: &'a str,
    cfg: &'a BalancedConfig,
    pos: usize,
    depth_brace: usize,
    depth_bracket: usize,
    depth_paren: usize,
    in_quote: Option<QuoteState>,
}

enum QuoteState {
    Ascii(char),
    /// Inside a paired-sentinel string. The next sentinel match closes.
    Paired,
}

impl<'a> Walker<'a> {
    fn new(text: &'a str, cfg: &'a BalancedConfig) -> Self {
        Self {
            text,
            cfg,
            pos: 0,
            depth_brace: 0,
            depth_bracket: 0,
            depth_paren: 0,
            in_quote: None,
        }
    }

    fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    fn at_top_level(&self) -> bool {
        self.depth_brace == 0
            && self.depth_bracket == 0
            && (!self.cfg.include_parens || self.depth_paren == 0)
            && self.in_quote.is_none()
    }

    /// Walk forward consuming bytes. Returns `Some(idx)` when a
    /// top-level separator is found. Returns `None` at end-of-input
    /// only if everything was balanced; otherwise `Err(message)`.
    fn advance_to_separator(&mut self) -> Result<Option<usize>, String> {
        self.find_char(self.cfg.separator)
    }

    /// Find the next top-level occurrence of `needle`. Drives the
    /// state machine forward; subsequent calls resume from where the
    /// last one stopped.
    fn find_char(&mut self, needle: char) -> Result<Option<usize>, String> {
        let bytes = self.text.as_bytes();
        while self.pos < bytes.len() {
            // String handling first — string content is opaque to the
            // bracket / separator logic.
            if let Some(state) = &self.in_quote {
                match state {
                    QuoteState::Ascii(open) => {
                        let ch = bytes[self.pos] as char;
                        if ch == '\\' && self.pos + 1 < bytes.len() {
                            self.pos += 2;
                            continue;
                        }
                        if ch == *open {
                            self.in_quote = None;
                        }
                        self.pos += 1;
                        continue;
                    }
                    QuoteState::Paired => {
                        if let StringQuote::PairedSentinel(s) = self.cfg.string_quote {
                            if self.text[self.pos..].starts_with(s) {
                                self.in_quote = None;
                                self.pos += s.len();
                                continue;
                            }
                        }
                        self.pos += 1;
                        continue;
                    }
                }
            }
            // String openers.
            match self.cfg.string_quote {
                StringQuote::Ascii => {
                    let ch = bytes[self.pos] as char;
                    if ch == '\'' || ch == '"' {
                        self.in_quote = Some(QuoteState::Ascii(ch));
                        self.pos += 1;
                        continue;
                    }
                }
                StringQuote::PairedSentinel(s) => {
                    if self.text[self.pos..].starts_with(s) {
                        self.in_quote = Some(QuoteState::Paired);
                        self.pos += s.len();
                        continue;
                    }
                }
            }
            // Bracket / separator dispatch.
            let ch = bytes[self.pos] as char;
            match ch {
                '{' => self.depth_brace += 1,
                '}' => self.depth_brace = self.depth_brace.saturating_sub(1),
                '[' => self.depth_bracket += 1,
                ']' => self.depth_bracket = self.depth_bracket.saturating_sub(1),
                '(' if self.cfg.include_parens => self.depth_paren += 1,
                ')' if self.cfg.include_parens => {
                    self.depth_paren = self.depth_paren.saturating_sub(1)
                }
                c if c == needle && self.at_top_level() => {
                    return Ok(Some(self.pos));
                }
                _ => {}
            }
            self.pos += 1;
        }
        if !self.at_top_level() {
            return Err(format!(
                "unterminated tool-call expression: `{}`",
                self.text
            ));
        }
        Ok(None)
    }
}
