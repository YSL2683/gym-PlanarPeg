"""
Pre-defined grid maps for the PlanarPeg environment.
"""

MAPS = {
    "default": [
        "WWWWWWWWWWW",
        "W....O....W",
        "W....O....W",
        "W....O....W",
        "W.S.....C.W",
        "W....O....W",
        "W....O....W",
        "W....O....W",
        "WWWWWWWWWWW"
    ],
    "empty": [
        "WWWWWWWWWWW",
        "W.........W",
        "W.........W",
        "W.........W",
        "W.S.....C.W",
        "W.........W",
        "W.........W",
        "W.........W",
        "WWWWWWWWWWW"
    ]
}

def get_map(map_name: str) -> list:
    """Returns the grid layout for the given map name."""
    if map_name not in MAPS:
        raise ValueError(f"Map '{map_name}' not found. Available maps: {list(MAPS.keys())}")
    return MAPS[map_name]
