"""The archive's own files, parsed here.

The relational ARFF has one trap and it is the reason this module exists rather than a library
call: an instance is a quoted field followed by a class value that may itself be quoted, so a
parser that finds the closing quote from the right end of the line reads the class value's quote
and swallows the label into the data. The fixtures below carry a quoted class value on purpose.

The rest of the tests cover what the archive actually contains and this reader has to survive:
``?`` for missing, a class attribute whose name varies per dataset, a ragged dataset padded to a
fixed inner width, and a ``.ts`` copy of the same data that disagrees with the ARFF about that
padding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.tsc.reader import (
    agree,
    finite_prefix,
    read_arff,
    read_ts,
    summarize,
)

ARFF_HEADER = """% a comment that is not UTF-8 safe in the archive: \xb5
@relation 'Probe'
@attribute relationalAtt relational
@attribute att0 numeric
@attribute att1 numeric
@attribute att2 numeric
@end relationalAtt
@attribute activity {'Standing',Running}
@data
"""

# Two dimensions per instance, three timepoints each, and the second instance's class value is
# quoted while the first one's is not. Both forms appear in the archive.
ARFF_BODY = """'1,2,3\\n4,5,6',Running
'7,?,9\\n10,11,12','Standing'
"""

TS_TEXT = """#a comment
@problemName Probe
@timeStamps false
@univariate false
@dimensions 2
@equalLength true
@seriesLength 3
@classLabel true Standing Running
@data
1,2,3:4,5,6:Running
7,?,9:10,11,12:Standing
"""


@pytest.fixture
def arff(tmp_path: Path) -> Path:
    """A two-instance relational ARFF with one quoted and one bare class value."""
    path = tmp_path / "Probe_TRAIN.arff"
    path.write_text(ARFF_HEADER + ARFF_BODY, encoding="latin-1")
    return path


def test_an_instance_is_channel_major_on_disk_and_time_major_in_memory(
    arff: Path,
) -> None:
    """A dimension is a row in the file and a column in the array.

    The reference preprocessing produces ``(length, dimensions)``, so a reader that skipped the
    transpose would train every model on a transposed dataset and never raise.
    """
    split = read_arff(arff)
    assert split.lengths == [3, 3]
    assert split.dimensions == 2
    assert np.array_equal(
        split.series[0], np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    )
    assert split.series[0].dtype == np.float64


def test_a_quoted_class_value_does_not_swallow_the_label(arff: Path) -> None:
    """The field closes at the first repeat of the opening quote, not the last quote on the line.

    Closing from the right would put ``,'Standing'`` inside the numeric field on the second
    instance and fail with a parse error on a file the archive really contains.
    """
    split = read_arff(arff)
    assert split.labels == ["Running", "Standing"]


def test_a_question_mark_is_nan_and_not_a_zero(arff: Path) -> None:
    """Missing values arrive as NaN so the pipeline can see them.

    Zero-filling would be an imputation this harness is not allowed to make: the reference does
    not impute, and a NaN row is what makes deduplication keep an instance.
    """
    split = read_arff(arff)
    assert np.isnan(split.series[1][1, 0])
    assert split.series[1][1, 1] == 11.0
    assert summarize(split)["missing"] == pytest.approx(1.0 / 12.0)


def test_the_ts_copy_agrees_with_the_arff(tmp_path: Path, arff: Path) -> None:
    """Two file formats, one dataset, elementwise equal with NaN counted as equal.

    This is the cross-check that the ARFF path is reading the archive and not a plausible
    rearrangement of it.
    """
    ts = tmp_path / "Probe_TRAIN.ts"
    ts.write_text(TS_TEXT, encoding="latin-1")
    assert agree(read_arff(arff), read_ts(ts))


def test_a_padded_instance_reports_its_real_length(tmp_path: Path) -> None:
    """The ARFF's fixed inner width pads a short instance, and the padding is measurable.

    Four of the archive's thirty datasets are ragged. The pad is trailing all-NaN timepoints, so
    :func:`finite_prefix` is how long the instance really is.
    """
    path = tmp_path / "Ragged_TRAIN.arff"
    path.write_text(ARFF_HEADER + "'1,2,?\\n4,5,?',Running\n", encoding="latin-1")
    item = read_arff(path).series[0]
    assert item.shape == (3, 2)
    assert finite_prefix(item) == 2


def test_a_malformed_file_is_refused_rather_than_half_read(tmp_path: Path) -> None:
    """A missing section, an unquoted instance or an undeclared class stops the read.

    Each of the three yields a partial dataset if tolerated, and a partial dataset trains.
    """
    cases = (
        ("has no @data section", ARFF_HEADER.split("@data")[0]),
        ("is not a quoted field", ARFF_HEADER + "1,2,3\\n4,5,6,Running\n"),
        ("is not one of", ARFF_HEADER + "'1,2,3\\n4,5,6',Walking\n"),
    )
    for position, (message, text) in enumerate(cases):
        path = tmp_path / f"Bad{position}_TRAIN.arff"
        path.write_text(text, encoding="latin-1")
        with pytest.raises(ValueError, match=message):
            read_arff(path)


def test_a_channel_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    """A row that does not fill the declared inner width stops the read.

    The declared width is the only statement the file makes about its own shape, so a row that
    disagrees with it means the separator was misread and every later instance is wrong too.
    """
    path = tmp_path / "Short_TRAIN.arff"
    path.write_text(ARFF_HEADER + "'1,2\\n4,5',Running\n", encoding="latin-1")
    with pytest.raises(ValueError, match="expected 3"):
        read_arff(path)


def test_a_timestamped_ts_file_is_refused(tmp_path: Path) -> None:
    """A format this reader has never seen is refused, not guessed at.

    The multivariate archive carries no timestamped file, so a parser for one would be untested
    code that silently accepts something else.
    """
    path = tmp_path / "Stamped_TRAIN.ts"
    path.write_text(TS_TEXT.replace("@timeStamps false", "@timeStamps true"), "latin-1")
    with pytest.raises(ValueError, match="timestamped"):
        read_ts(path)
