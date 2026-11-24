"""
text notation for the rotations
the rotation is drawn for the teams perspective facing the net. 

O - O - S
O - O - O

the layout comes from CourtModel.slot.grid, so this model adds no geometry of its own.
this way of drawing it is for checking the rotation for myself and not for the code. Nothing stores a rotation in this form because a rotational state is fully described by the setter's slot

"""

from __future__ import annotations

from src.courtmodel.model import SLOT_COUNT, CourtModel

SETTER_MARK = "S"
OTHER_MARK = "O"
CELL_SEPARATOR = " - "
ROW_SEPARATOR = "\n"

def render_rotation(rotation: int, court: CourtModel | None = None) -> str:
    """Return the rotational state with the setter in rotzation as text"""
    court = CourtModel() if court is None else court
    setter = court.setter_slot(rotation)
    rows = [
        CELL_SEPARATOR.join(
            SETTER_MARK if slot == setter else OTHER_MARK for slot in row
        )
        for row in court.slot_grid()
    ]
    return ROW_SEPARATOR.join(rows)

def parse_rotation(text: str, court: CourtModel| None = None) -> int:
    """return the setters slot from a rotational state written as text"""
    court = CourtModel() if court is None else court
    grid = court.slot_grid()

    rows = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if len(rows) != len(grid):
        raise ValueError(
            f"a rotation needs {len(grid)} rows, front row first, got {len(rows)}"
        )

    setters: list[int] = []
    for row_index, (row, slots) in enumerate(zip(rows, grid)):
        cells = [cell for cell in row.replace("-", " ").split() if cell]
        if len(cells) != len(slots):
            raise ValueError(
                f"row {row_index + 1} needs {len(slots)} cells, got {len(cells)}: "
                f"{row!r}"
            )
        for cell, slot in zip(cells, slots):
            mark = cell.upper()
            if mark == SETTER_MARK:
                setters.append(slot)
            elif mark != OTHER_MARK:
                raise ValueError(
                    f"cell {cell!r} is neither {SETTER_MARK!r} nor {OTHER_MARK!r}"
                )
    if len(setters) != 1:
        raise ValueError(f"a 5-1 rotation has exactly one setter, found {len(setters)}")
    return setters[0]

def main() -> int:
    """print every rotation. entry point for the cli"""
    court = CourtModel()
    for rotation in range(1, SLOT_COUNT + 1):
        row = "front" if court.is_front_row_setter(rotation) else "back"
        print(f"setter in slot {rotation} ({row} row):")
        print(render_rotation(rotation, court))
        print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    