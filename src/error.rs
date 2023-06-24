use core::fmt;

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

pub struct Error {
    repr: Repr,
}

pub(crate) enum Repr {
    Other(String),
}

impl From<String> for Repr {
    fn from(value: String) -> Self {
        Self::Other(value)
    }
}
impl<'a> From<&'a str> for Repr {
    fn from(value: &'a str) -> Self {
        Self::Other(value.into())
    }
}

impl Error {
    pub(crate) fn from(e: impl Into<Repr>) -> Self {
        Self { repr: e.into() }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.repr {
            Repr::Other(s) => s.fmt(f),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.repr {
            Repr::Other(s) => s.fmt(f),
        }
    }
}

impl std::error::Error for Error {}
