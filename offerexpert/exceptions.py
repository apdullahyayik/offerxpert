"""Module for exceptions."""


class OfferExpertError(Exception):
    """General offer expert error."""


class OfferExpertEnvironmentError(OfferExpertError):
    """Missing environment error."""
