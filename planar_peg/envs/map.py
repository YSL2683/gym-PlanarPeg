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
    "ood_goal_position": [
            "WWWWWWWWWWW",
            "W....O....W",
            "W....O....W",
            "W....O..C.W",
            "W.S.......W",
            "W....O....W",
            "W....O....W",
            "W....O....W",
            "WWWWWWWWWWW"
        ],
    "ood_start_position": [
            "WWWWWWWWWWW",
            "W....O....W",
            "W....O....W",
            "W.S..O....W",
            "W.......C.W",
            "W....O....W",
            "W....O....W",
            "W....O....W",
            "WWWWWWWWWWW"
        ],
    "ood_obstacle_position": [
            "WWWWWWWWWWW",
            "W....O....W",
            "W....O....W",
            "W.........W",
            "W.S..O..C.W",
            "W....O....W",
            "W....O....W",
            "W....O....W",
            "WWWWWWWWWWW"
        ]       

}

def get_map(map_name: str) -> list:
    """Returns the grid layout for the given map name."""
    if map_name not in MAPS:
        raise ValueError(f"Map '{map_name}' not found. Available maps: {list(MAPS.keys())}")
    return MAPS[map_name]
