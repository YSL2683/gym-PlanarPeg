import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional

class MazeModel:
    """
    MazeModel parses a text-based grid map and generates a MuJoCo XML configuration.
    It loads the base_peg.xml template containing the agent definition and adds 
    static walls and C-shaped goal slots dynamically based on the grid layout.
    """
    def __init__(self, grid: List[str], cell_size: float = 0.2, base_xml_path: Optional[str] = None):
        """
        Initializes the MazeModel.

        Args:
            grid (List[str]): A list of strings representing the grid layout.
                'W' or '#': Wall
                'S' or 'P': Agent start position
                'G' or 'C': Goal position (C-shaped slot opening to the left)
                '.': Empty space
            cell_size (float): The physical size of a single grid cell in meters.
            base_xml_path (Optional[str]): Absolute path to the base_peg.xml template.
                If None, it infers the path relative to this file.
        """
        self.grid = grid
        self.cell_size = cell_size
        
        if base_xml_path is None:
            # Locate base_peg.xml relative to this file's location
            base_xml_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'assets', 'base_peg.xml'
            )
        self.base_xml_path = base_xml_path
        
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        
        # Offsets to center the grid at the coordinate system origin (0, 0)
        self.x_offset = (self.width - 1) * self.cell_size / 2.0
        self.y_offset = (self.height - 1) * self.cell_size / 2.0
        
        self.start_pos: Tuple[float, float] = (0.0, 0.0)
        self.goal_pos: Tuple[float, float] = (0.5, 0.0)
        
        self._parse_grid()

    def grid_to_xy(self, r: int, c: int) -> Tuple[float, float]:
        """
        Converts grid row and column index to physical (X, Y) coordinates.
        The origin (0, 0) is placed at the center of the grid map.
        
        Args:
            r (int): Grid row index.
            c (int): Grid column index.

        Returns:
            Tuple[float, float]: Physical X and Y coordinates.
        """
        x = c * self.cell_size - self.x_offset
        y = (self.height - 1 - r) * self.cell_size - self.y_offset
        return x, y

    def _parse_grid(self) -> None:
        """Parses the grid to extract agent start and goal positions."""
        start_found = False
        goal_found = False
        
        for r in range(self.height):
            for c in range(self.width):
                char = self.grid[r][c]
                if char in ['S', 'P']:
                    self.start_pos = self.grid_to_xy(r, c)
                    start_found = True
                elif char in ['G', 'C']:
                    self.goal_pos = self.grid_to_xy(r, c)
                    goal_found = True
                    
        if not start_found:
            # Default fallback if 'S'/'P' is missing
            self.start_pos = self.grid_to_xy(self.height // 2, 1)
        if not goal_found:
            # Default fallback if 'G'/'C' is missing
            self.goal_pos = self.grid_to_xy(self.height // 2, self.width - 2)

    def generate_xml_string(self) -> str:
        """
        Loads the base XML template, updates the starting coordinates of the agent
        and its corresponding mocap target body, parses the grid layout to append
        static walls and a C-shaped goal slot, and returns the full XML as a string.

        Returns:
            str: XML string compatible with MuJoCo.
        """
        if not os.path.exists(self.base_xml_path):
            raise FileNotFoundError(f"Base XML template not found at: {self.base_xml_path}")
            
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        
        worldbody = root.find('worldbody')
        if worldbody is None:
            raise ValueError("worldbody tag not found in the base XML template")

        # Update the start positions of both mocap and agent body
        mocap_body = worldbody.find("./body[@name='mocap']")
        if mocap_body is not None:
            mocap_body.set('pos', f"{self.start_pos[0]} {self.start_pos[1]} 0.025")
            
        agent_body = worldbody.find("./body[@name='agent']")
        if agent_body is not None:
            agent_body.set('pos', f"{self.start_pos[0]} {self.start_pos[1]} 0.025")

        # Build map structures
        for r in range(self.height):
            for c in range(self.width):
                char = self.grid[r][c]
                x, y = self.grid_to_xy(r, c)
                
                if char in ['W', '#']:
                    # Create a static body for each wall cell
                    wall = ET.SubElement(worldbody, 'body', {
                        'name': f'wall_{r}_{c}',
                        'pos': f'{x} {y} 0.05'
                    })
                    half_w = self.cell_size / 2.0
                    ET.SubElement(wall, 'geom', {
                        'name': f'wall_geom_{r}_{c}',
                        'type': 'box',
                        'size': f'{half_w} {half_w} 0.05',
                        'rgba': '0.3 0.3 0.3 1.0',
                        'condim': '3'
                    })
                    
                elif char in ['O']:
                    # Create a static body for each obstacle cell (Blue color)
                    obstacle = ET.SubElement(worldbody, 'body', {
                        'name': f'obstacle_{r}_{c}',
                        'pos': f'{x} {y} 0.05'
                    })
                    half_w = self.cell_size / 2.0
                    ET.SubElement(obstacle, 'geom', {
                        'name': f'obstacle_geom_{r}_{c}',
                        'type': 'box',
                        'size': f'{half_w} {half_w} 0.05',
                        'rgba': '0.2 0.4 0.8 1.0', # Blue color
                        'condim': '3'
                    })
                    
                elif char in ['G', 'C']:
                    # Create the Goal slot.
                    # It is modeled as a C-shaped pocket opening to the left (-X direction).
                    # It comprises three wall-like geoms (back, top, and bottom) within a single body.
                    goal = ET.SubElement(worldbody, 'body', {
                        'name': 'goal_pocket',
                        'pos': f'{x} {y} 0.05'
                    })
                    
                    half_w = self.cell_size / 2.0
                    thickness = 0.02
                    half_thickness = thickness / 2.0
                    
                    # 1. Back wall (closing the pocket on the right side)
                    ET.SubElement(goal, 'geom', {
                        'name': 'goal_back_geom',
                        'type': 'box',
                        'pos': f'{half_w - half_thickness} 0 0',
                        'size': f'{half_thickness} {half_w} 0.05',
                        'rgba': '0.1 0.7 0.2 0.6',
                        'condim': '3'
                    })
                    
                    # 2. Top wall (horizontal barrier at the upper bound)
                    ET.SubElement(goal, 'geom', {
                        'name': 'goal_top_geom',
                        'type': 'box',
                        'pos': f'{-half_thickness} {half_w - half_thickness} 0',
                        'size': f'{half_w - half_thickness} {half_thickness} 0.05',
                        'rgba': '0.1 0.7 0.2 0.6',
                        'condim': '3'
                    })
                    
                    # 3. Bottom wall (horizontal barrier at the lower bound)
                    ET.SubElement(goal, 'geom', {
                        'name': 'goal_bottom_geom',
                        'type': 'box',
                        'pos': f'{-half_thickness} {-half_w + half_thickness} 0',
                        'size': f'{half_w - half_thickness} {half_thickness} 0.05',
                        'rgba': '0.1 0.7 0.2 0.6',
                        'condim': '3'
                    })
                    
        return ET.tostring(root, encoding='utf-8').decode('utf-8')
