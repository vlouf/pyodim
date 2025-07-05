"""
Test cases for pyodim exceptions.
"""

import pytest
from pyodim.exceptions import (
    PyOdimError,
    FileNotFoundError,
    InvalidFileFormatError,
    InvalidSliceError,
    MetadataError,
    CoordinateError,
    DataValidationError,
    NyquistConsistencyError,
    FieldNotFoundError,
    H5FileError,
)


class TestExceptionHierarchy:
    """Test the exception hierarchy."""

    def test_base_exception(self):
        """Test the base PyOdimError exception."""
        exc = PyOdimError("Test message")
        assert str(exc) == "Test message"
        assert isinstance(exc, Exception)

    def test_file_not_found_error(self):
        """Test FileNotFoundError inherits from PyOdimError."""
        exc = FileNotFoundError("File not found")
        assert isinstance(exc, PyOdimError)
        assert isinstance(exc, Exception)
        assert str(exc) == "File not found"

    def test_invalid_file_format_error(self):
        """Test InvalidFileFormatError inherits from PyOdimError."""
        exc = InvalidFileFormatError("Invalid format")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Invalid format"

    def test_invalid_slice_error(self):
        """Test InvalidSliceError inherits from PyOdimError."""
        exc = InvalidSliceError("Invalid slice")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Invalid slice"

    def test_metadata_error(self):
        """Test MetadataError inherits from PyOdimError."""
        exc = MetadataError("Metadata error")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Metadata error"

    def test_coordinate_error(self):
        """Test CoordinateError inherits from PyOdimError."""
        exc = CoordinateError("Coordinate error")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Coordinate error"

    def test_data_validation_error(self):
        """Test DataValidationError inherits from PyOdimError."""
        exc = DataValidationError("Data validation error")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Data validation error"

    def test_nyquist_consistency_error(self):
        """Test NyquistConsistencyError inherits from PyOdimError."""
        exc = NyquistConsistencyError("Nyquist error")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Nyquist error"

    def test_field_not_found_error(self):
        """Test FieldNotFoundError inherits from PyOdimError."""
        exc = FieldNotFoundError("Field not found")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "Field not found"

    def test_h5_file_error(self):
        """Test H5FileError inherits from PyOdimError."""
        exc = H5FileError("H5 file error")
        assert isinstance(exc, PyOdimError)
        assert str(exc) == "H5 file error"


class TestExceptionUsage:
    """Test practical usage of exceptions."""

    def test_exception_with_context(self):
        """Test exceptions with additional context."""
        try:
            raise FileNotFoundError("File 'test.h5' not found in directory '/data'")
        except FileNotFoundError as e:
            assert "test.h5" in str(e)
            assert "/data" in str(e)

    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise H5FileError("H5 error occurred") from e
        except H5FileError as e:
            assert str(e) == "H5 error occurred"
            assert isinstance(e.__cause__, ValueError)

    def test_catching_base_exception(self):
        """Test catching all pyodim exceptions with base class."""
        exceptions_to_test = [
            FileNotFoundError("test"),
            InvalidFileFormatError("test"),
            InvalidSliceError("test"),
            MetadataError("test"),
            CoordinateError("test"),
            DataValidationError("test"),
            NyquistConsistencyError("test"),
            FieldNotFoundError("test"),
            H5FileError("test"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except PyOdimError:
                pass  # Should catch all pyodim exceptions
            else:
                pytest.fail(f"Failed to catch {type(exc).__name__} with PyOdimError")
