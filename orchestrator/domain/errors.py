class DomainRuleViolation(RuntimeError):
    """Raised when a domain invariant is violated."""


class InvalidTransitionError(DomainRuleViolation):
    """Raised when an invalid run status transition is attempted."""

