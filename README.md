Group Concept Mapping
=====================

This implements Jackson and Trochim's "Concept Mapping" algorithm [1].
This is more commonly known as "group concept mapping", to avoid confusion
with other things also called "concept mapping".

1.  Kristin M. Jackson and William M. K. Trochim,
    _"Concept Mapping as an Alternative Approach for the Analysis
    of Open-Ended Survey Responses"_, Organizational Research Methods,
    Vol. 5 No. 4, October 2002 (pp. 307-336)

This was written mainly to process the data for S's papers, not as a general tool,
so there isn't a lot of documentation :-(

The input is a .xlsx file containing all statements (vertically) and all labeled groups
(horizontally); cell (i, j) contains "1" if the i'th statement is included in the j'th
group, and "0" otherwise:

            lbl1    lbl2    lbl3    ...
    stmt1   0       1       0       ...
    stmt2   1       1       0       ...
    stmt3   0       1       1       ...
    ...


## License

This work is released into the public domain, under the terms of the
[CC0 license](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
You can copy, modify, distribute and perform the work, even for commercial purposes,
all without asking permission. See the file 'COPYING' for more information.


## NO WARRANTY

BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

