import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("carcar")

FPS = 60

# Colors for rendering different elements
BG = (0, 0, 0)        # Background color
ROAD = (70, 70, 70)    # Road color
WHITE = (255, 255, 255) # Ray color
GREEN = (0, 255, 0)    # Checkpoint color


CHECKPOINTS = [
        ((545, 150), (545, 250)),
        ((640, 263), (715, 185)),
        ((620, 290), (730, 340)),
        ((610, 430), (690, 379)),
        ((710, 400), (700, 500)),
        ((830, 450), (770, 525)),
        ((800, 600), (900, 600)),
        ((850, 770), (770, 640)),
        ((700, 700), (700, 800)),
        ((550, 770), (620, 640)),
        ((500, 600), (600, 600)),
        ((475, 475), (475, 575)),
        ((350, 460), (300, 560)),
        ((185, 465), (295, 405)),
        ((175, 375), (275, 375)),
        ((185, 260), (295, 305)),
        ((350, 150), (350, 250))
    ]
# Car Class
class CarGameAI:
    """
    CarGameAI class represents an AI-controlled car with capabilities to move, detect the road,
    pass checkpoints, and reset upon going off-track. Includes methods for rendering,
    raycasting for obstacle detection, and handling checkpoints.
    """
    def __init__(self, x, y):
        self.start_x, self.start_y = WIDTH // 2, HEIGHT // 2 - 200
        self.x = x
        self.y = y
        self.angle = 0  # Initial direction in degrees
        self.velocity = 0  # Initial velocity
        self.trail = []  # Store trail positions for visual effect

        # Road detection and checkpoint setup
        self.road_surface = get_track_surface()  # Load the track surface
        self.road_mask = pygame.mask.from_threshold(self.road_surface, ROAD, (1, 1, 1, 255))  # Road mask for collisions
        self.checkpoints = CHECKPOINTS
        self.checkpoint_passed = []  # Track passed checkpoints
        self.checkpoint_index = 0

        self.ray_distances = []  # Store distances for rays in all directions

        # Load and scale car image
        self.car_image = pygame.image.load("car.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (20, 40))
        self.car_image = pygame.transform.rotate(self.car_image, 90)
        self.rect = self.car_image.get_rect(center=(self.x, self.y))

        # Game metrics
        self.time = 0
        self.score = 0
        self.rewards = 0

        self.reset()  # Reset the car's position and variables
        self.step([0, 0, 0, 0])  # Initialize with a no-action step

    def move(self, action):
        """
        Move the car based on the input action array, adjusting velocity and angle accordingly.

        Args:
            action (list): A list where each element represents an action (0: forward, 1: backward,
                           2: turn left, 3: turn right).
        """
        acceleration = 0.1  # Acceleration factor

        # Handle forward and backward movement
        moved = False
        if bool(action[0]):
            self.velocity = min(self.velocity + acceleration, 12)  # Max speed = 12
            moved = True
        if bool(action[1]):
            self.velocity = max(self.velocity - acceleration, -6)  # Min speed = -6
            moved = True

        # Handle rotation (left/right)
        if bool(action[2]):
            self.angle += 2
        if bool(action[3]):
            self.angle -= 2

        # Update car position based on direction and velocity
        radians = math.radians(self.angle)
        self.x += self.velocity * math.cos(radians)
        self.y -= self.velocity * math.sin(radians)
        self.rect = self.car_image.get_rect(center=(self.x, self.y))  # Update position rectangle

    def cast_ray(self, angle_offset, screen, road_mask):
        """
        Cast a ray to measure the distance to the road boundary in a specific direction.

        Args:
            angle_offset (float): The angle offset from the car's direction to cast the ray.
            screen (pygame.Surface): The Pygame screen to draw on.
            road_mask (pygame.mask.Mask): The mask representing the road boundaries.

        Returns:
            int: Distance from the car to the road boundary, or max distance if boundary not reached.
        """
        max_distance = 500  # Maximum ray length
        step_size = 1  # How far the ray moves each step

        radians = math.radians(self.angle + angle_offset)
        ray_x, ray_y = self.x, self.y

        for distance in range(0, max_distance, step_size):
            ray_x += step_size * math.cos(radians)
            ray_y -= step_size * math.sin(radians)

            # Return distance if road boundary is hit
            try:
                if not road_mask.get_at((int(ray_x), int(ray_y))):
                    pygame.draw.line(screen, WHITE, (self.x, self.y), (ray_x, ray_y), 2)
                    return distance
            except IndexError:
                pass

        return max_distance  # If no boundary is hit, return max distance

    def get_distances(self, screen, road_mask):
        """
        Retrieve distances in various directions to detect obstacles around the car.

        Args:
            screen (pygame.Surface): The Pygame screen to draw on.
            road_mask (pygame.mask.Mask): The mask representing the road boundaries.

        Returns:
            list: A list of distances in various directions around the car.
        """
        distances = []
        for angle_offset in [-135, -90, -45, -22.5, 0, 22.5, 45, 90, 135]:
            distance = self.cast_ray(angle_offset, screen, road_mask)
            distances.append(distance)
        return distances

    def draw(self, screen):
        """
        Draw the car and additional visual elements like the trail and checkpoint lines.

        Args:
            screen (pygame.Surface): The Pygame screen to draw on.
        """
        self.update_trail()
        self.draw_trail(screen)
        self.draw_data(screen)

        # Draw checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.line(WIN, GREEN, checkpoint[0], checkpoint[1], 2)

        # Rotate and render car image
        rotated_car = pygame.transform.rotate(self.car_image, self.angle)
        new_rect = rotated_car.get_rect(center=self.rect.center)
        screen.blit(rotated_car, new_rect.topleft)

        # Cast rays and update ray distances
        self.ray_distances = self.get_distances(screen, self.road_mask)

    def check_on_track(self, road_mask):
        """
        Check if the car is within the road boundaries based on the mask.

        Args:
            road_mask (pygame.mask.Mask): The mask representing the road boundaries.

        Returns:
            bool: True if the car is on the track, False otherwise.
        """
        car_pos = (int(self.x), int(self.y))
        if 0 <= car_pos[0] < road_mask.get_size()[0] and 0 <= car_pos[1] < road_mask.get_size()[1]:
            return road_mask.get_at(car_pos) == 1
        return False

    def reset(self):
        """
        Reset the car's position and parameters upon collision or completion.
        """
        self.x, self.y = self.start_x, self.start_y
        self.angle = 0
        self.velocity = 0
        self.clock = pygame.time.Clock()
        self.time = 0
        self.checkpoint_index = 0

    def add_to_trail(self):
        """
        Add a point to the trail for rendering car movement history.
        """
        trail_lifetime = 100
        self.trail.append({"pos": (self.x, self.y), "angle": self.angle, "lifetime": trail_lifetime, "alpha": 255})

    def update_trail(self):
        """
        Update the trail by fading and removing old points.
        """
        for trail in self.trail[:]:
            trail['lifetime'] -= 1
            trail['alpha'] = max(0, int(255 * (trail['lifetime'] / 100)))
            if trail['lifetime'] <= 0:
                self.trail.remove(trail)

    def draw_trail(self, screen):
        """
        Draw the fading trail of the car on the screen.

        Args:
            screen (pygame.Surface): The Pygame screen to draw on.
        """
        for trail in self.trail:
            trail_pos = trail['pos']
            trail_radius = 10
            trail_surface = pygame.Surface((trail_radius * 2, trail_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (0, 0, 0, trail['alpha']), (trail_radius, trail_radius), trail_radius)
            screen.blit(trail_surface, (trail_pos[0] - trail_radius, trail_pos[1] - trail_radius))

    def draw_data(self, screen):
        """
        Display game data (time, speed, checkpoint, rewards) on the screen.

        Args:
            screen (pygame.Surface): The Pygame screen to draw on.
        """
        elapsed_time = self.time / 60
        font = pygame.font.Font(None, 36)
        timer_surface = font.render(f"Time: {elapsed_time:.2f} seconds", True, WHITE)
        screen.blit(timer_surface, (10, 10))

        speed_surface = font.render(f"Speed: {self.velocity:.2f} m/s", True, WHITE)
        screen.blit(speed_surface, (10, 60))

        checkpoint_dist_surface = font.render(f"Checkpoint: {self.checkpoint_index}", True, WHITE)
        screen.blit(checkpoint_dist_surface, (10, 110))

        reward_surface = font.render(f"Rewards: {self.rewards:.2f}", True, WHITE)
        screen.blit(reward_surface, (10, 160))

    def check_checkpoint(self, checkpoint):
        """
        Check if the car has passed a checkpoint and return the checkpoint if passed.

        Args:
            checkpoint (tuple): A tuple of two points representing the checkpoint line.

        Returns:
            tuple or None: The checkpoint if passed, None otherwise.
        """
        car_rect = self.car_image.get_rect(center=(self.x, self.y))
        if car_rect.clipline(checkpoint[0], checkpoint[1]):
            return checkpoint
        return None

    def distance_to_next_checkpoint(self):
        """
        Calculate the distance to the next checkpoint.

        Returns:
            int: The distance to the next checkpoint, or 10000 if there are no more checkpoints.
        """
        if self.checkpoint_index < len(self.checkpoints):
            checkpoint = self.checkpoints[self.checkpoint_index]
            checkpoint_center = checkpoint.center  # Get the center of the checkpoint rectangle
            distance = math.sqrt((self.x - checkpoint_center[0]) ** 2 + (self.y - checkpoint_center[1]) ** 2)
            return int(distance)
        return 10000  # Return 10000 if no more checkpoints

    def velocity_towards_next_checkpoint(self):
        """Calculates the car's velocity projection towards the next checkpoint.

        Returns:
            float: The scalar projection of the car's velocity in the direction of the next checkpoint.
            Returns 0 if there are no more checkpoints.
        """
        if self.checkpoint_index >= len(self.checkpoints):
            return 0  # No more checkpoints to go to

        # Get the next checkpoint's center position
        next_checkpoint = self.checkpoints[self.checkpoint_index]
        checkpoint_center = get_line_center(next_checkpoint[0], next_checkpoint[1])

        # Calculate the vector from the car to the checkpoint
        dx = checkpoint_center[0] - self.x
        dy = checkpoint_center[1] - self.y

        # Calculate distance to avoid division by zero
        distance_to_checkpoint = math.sqrt(dx ** 2 + dy ** 2)
        if distance_to_checkpoint == 0:
            return 0

        # Normalize the direction vector to the checkpoint
        direction_to_checkpoint = (dx / distance_to_checkpoint, dy / distance_to_checkpoint)

        # Car's current velocity vector
        car_velocity_vector = (self.velocity * math.cos(math.radians(self.angle)),
                               -self.velocity * math.sin(math.radians(self.angle)))

        # Project the car's velocity onto the checkpoint direction vector
        velocity_towards_checkpoint = (car_velocity_vector[0] * direction_to_checkpoint[0] +
                                       car_velocity_vector[1] * direction_to_checkpoint[1])

        return velocity_towards_checkpoint

    def step(self, action):
        """Performs a single step in the environment based on the action taken.

        Args:
            action (any): The action that affects the car's movement.

        Returns:
            tuple: Contains the current state (velocity and ray distances),
                   the reward earned in this step,
                   a list of checkpoints passed,
                   and a boolean indicating if the episode has ended.
        """
        self.time += 1  # Increment time
        self.move(action)  # Move the car according to the action

        # Check for quit events to allow the program to close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward = 0  # Initialize reward

        # Check if the car has passed the next checkpoint
        checkpoint = self.check_checkpoint(self.checkpoints[self.checkpoint_index])
        if checkpoint and checkpoint not in self.checkpoint_passed:
            self.checkpoint_passed.append(checkpoint)  # Mark checkpoint as passed
            reward = 100  # Reward for passing a checkpoint
            self.score += 1
            self.checkpoint_index += 1  # Move to the next checkpoint
            print('Checkpoint ' + str(self.checkpoint_index) + ' hit!')

        # Check if the car is off-road
        if not self.check_on_track(self.road_mask):
            self.reset()  # Reset if off-road
            self.score = 0
            print("wall hit!")
            self.rewards += reward
            pygame.display.update()
            self.clock.tick(FPS)
            return [self.velocity] + self.ray_distances, reward, self.checkpoint_passed, True
        elif len(self.checkpoints) == len(self.checkpoint_passed):
            # If all checkpoints passed, end the episode with a high reward
            self.reset()
            reward = 1000
            self.score = 0
            print("track done!")
            self.rewards += reward
            pygame.display.update()
            self.clock.tick(FPS)
            return [self.velocity] + self.ray_distances, reward, self.checkpoint_passed, True

        # Render the current frame
        WIN.fill(BG)
        WIN.blit(self.road_surface, (0, 0))
        self.draw(WIN)

        # Reward for moving towards the next checkpoint
        reward += (0.15 if self.velocity_towards_next_checkpoint() > 0 and self.velocity > 0.25 else -0.15)
        self.rewards += reward  # Accumulate rewards

        pygame.display.update()
        self.clock.tick(FPS)

        return [self.velocity] + self.ray_distances, reward, self.checkpoint_passed, False

def get_line_center(point1, point2):
    """Calculates the midpoint of a line segment defined by two points.

    Args:
        point1 (tuple): The first point of the line segment.
        point2 (tuple): The second point of the line segment.

    Returns:
        tuple: The (x, y) coordinates of the midpoint.
    """
    x_center = (point1[0] + point2[0]) / 2
    y_center = (point1[1] + point2[1]) / 2
    return x_center, y_center

def get_track_surface():
    """Creates a surface for the track with predefined curves and lines.

    Returns:
        pygame.Surface: A surface object representing the road track.
    """
    road_surface = pygame.Surface((WIDTH, HEIGHT))
    for i in range(2):  # Draws the road track and curves
        pygame.draw.rect(road_surface, ROAD, pygame.Rect(WIDTH // 2 - 125 + i, HEIGHT // 2 - 250, 250, 100))
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 + i, HEIGHT // 2 - 250, 250, 250), -math.pi / 3, math.pi / 2, 100)
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 + 90 + i, HEIGHT // 2 - 150, 250, 250), 2 * math.pi / 3, 3 * math.pi / 2, 100)
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 + i, HEIGHT // 2, 400, 400), -math.pi, math.pi / 2, 100)
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 - 150 + i, HEIGHT // 2 + 75, 250, 250), 0, math.pi / 2, 100)
        pygame.draw.rect(road_surface, ROAD, pygame.Rect(WIDTH // 2 - 125 + i, HEIGHT // 2 + 75, 100, 100))
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 - 325 + i, HEIGHT // 2 - 250, 425, 425), math.pi / 2, 3 * math.pi / 2, 100)
    return road_surface