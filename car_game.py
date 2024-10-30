import pygame
import sys
import math
from enum import Enum
# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("carcar")

FPS = 60

# Colors
BG = (255, 120, 120)
ROAD = (70, 70, 70)
WHITE = (255, 255, 255)

class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DRIFT = 5

# Car Class
class CarGame:
    def __init__(self, x, y):
        self.start_x, self.start_y = WIDTH // 2, HEIGHT // 2 - 200

        self.x = x
        self.y = y
        self.angle = 0  # direction in degrees
        self.velocity = 0
        self.is_drifting = False  # Add drift state
        self.trail = []  # Store trail positions
        self.drift_angle = 0  # Angle of movement during drift

        # Load the car image and scale it down
        self.car_image = pygame.image.load("car.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (20, 40))  # Resize to 70x50
        self.car_image = pygame.transform.rotate(self.car_image, 90)

        self.rect = self.car_image.get_rect(center=(self.x, self.y))

        self.timer = 0

    def move(self):
        keys = pygame.key.get_pressed()

        # Check if drifting (spacebar is pressed)
        self.is_drifting = keys[pygame.K_SPACE]

        acceleration = 0.1 if not self.is_drifting else 0.025

        # Adjust velocity and rotation during drifting

        if self.is_drifting:
            drift_angle_adjustment = 50  # degrees
            # Reduced rotation during drifting
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.angle += 4  # More rotation while drifting
                self.drift_angle = self.angle + drift_angle_adjustment  # Angle left
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.angle -= 4
                self.drift_angle = self.angle - drift_angle_adjustment  # Angle right
            if not ((keys[pygame.K_LEFT] or keys[pygame.K_a]) or (keys[pygame.K_RIGHT] or keys[pygame.K_d])):
                self.drift_angle = self.angle
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.velocity += acceleration
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.velocity = max(self.velocity - acceleration, -6)

            if self.velocity > 0:
                self.velocity -= 0.05
            elif self.velocity < 0:
                self.velocity += 0.05


            self.add_to_trail()


        else:
            moved = False
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.velocity = min(self.velocity + acceleration, 12)  # Max speed = 12
                moved = True
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.velocity = max(self.velocity - acceleration, -6)  # Min speed = -12
                moved = True

            if not moved and self.velocity > 0:
                self.velocity -= acceleration * 1/2 * (1 if self.velocity > 0 else -1)
            # Normal driving controls
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.angle += 2  # Regular rotation
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.angle -= 2

        # Move the car in the direction it's facing
        radians = math.radians(self.angle)
        self.x += self.velocity * math.cos(radians)
        self.y -= self.velocity * math.sin(radians)  # Y decreases upward in Pygame

        # Update the car's rectangle position
        self.rect = self.car_image.get_rect(center=(self.x, self.y))

    def cast_ray(self, angle_offset, screen, road_mask):
        """Cast a ray in a given direction and calculate the distance to the road boundary."""
        max_distance = 500  # Maximum ray length
        step_size = 1  # How far the ray moves each step

        radians = math.radians(self.angle + angle_offset)  # Adjust the ray's angle
        ray_x, ray_y = self.x, self.y

        for distance in range(0, max_distance, step_size):
            ray_x += step_size * math.cos(radians)
            ray_y -= step_size * math.sin(radians)  # Y decreases upwards

            # If the ray hits the road boundary, return the distance
            try:
                if not road_mask.get_at((int(ray_x), int(ray_y))):
                    pygame.draw.line(screen, WHITE, (self.x, self.y), (ray_x, ray_y), 2)  # Draw the ray
                    return distance
            except IndexError:
                pass

        return max_distance  # If no boundary is hit, return max distance

    def get_distances(self, screen, road_mask):
        """Get distances to the road boundaries in multiple directions."""
        distances = []
        for angle_offset in [-135, -90, -45, -22.5, 0, 22.5, 45, 90, 135]:  # Cast rays in different directions
            distance = self.cast_ray(angle_offset, screen, road_mask)
            distances.append(distance)

        return distances

    def draw(self, screen, road_mask):
        # Rotate the car image
        if self.is_drifting:
            rotated_car = pygame.transform.rotate(self.car_image, self.drift_angle)
        else:
            rotated_car = pygame.transform.rotate(self.car_image, self.angle)

        # Get the new rectangle of the rotated image
        new_rect = rotated_car.get_rect(center=self.rect.center)

        # Draw the rotated car onto the screen
        screen.blit(rotated_car, new_rect.topleft)

        distances = self.get_distances(screen, road_mask)

    def check_on_track(self):
        """Check if the car is within road boundaries."""
        road_color = ROAD  # The road color
        # Get the color of the pixel where the car's center is
        car_center_color = WIN.get_at((int(self.x), int(self.y)))

        # If the car is not on the road color, return False
        return car_center_color == road_color

    def reset(self):
        """Resets the car to its starting position."""
        self.x = self.start_x
        self.y = self.start_y
        self.angle = 0  # Reset angle
        self.velocity = 0  # Stop the car
        self.timer = 0

    def add_to_trail(self):
        """Adds the current position and angle to the trail list with a lifetime."""
        trail_lifetime = 100  # How long the trail lasts (frames)
        self.trail.append({"pos": (self.x, self.y), "angle": self.angle, "lifetime": trail_lifetime, "alpha": 255})

    def update_trail(self):
        """Updates the trail list by reducing the lifetime and adjusting the alpha for fading."""
        for trail in self.trail[:]:
            trail['lifetime'] -= 1
            trail['alpha'] = max(0, int(255 * (trail['lifetime'] / 100)))  # Adjust alpha based on lifetime
            if trail['lifetime'] <= 0:
                self.trail.remove(trail)

    def draw_trail(self, screen):
        """Draws the trail behind the car with fading effect."""
        for trail in self.trail:
            trail_pos = trail['pos']
            trail_radius = 10  # Size of the trail

            # Create a surface for the trail
            trail_surface = pygame.Surface((trail_radius * 2, trail_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (0, 0, 0, trail['alpha']), (trail_radius, trail_radius), trail_radius)

            # Calculate the coordinates for drawing the trail
            screen.blit(trail_surface, (trail_pos[0] - trail_radius, trail_pos[1] - trail_radius))

    def update_timer(self, time):
        self.timer += time

    def draw_timer(self, screen):
        elapsed_time = self.timer / 1000
        font = pygame.font.Font(None, 36)
        timer_surface = font.render(f"Time: {elapsed_time:.2f} seconds", True, WHITE)
        screen.blit(timer_surface, (10, 10))

def draw_track():
    road_surface = pygame.Surface((WIDTH, HEIGHT))
    road_surface.fill(BG)
    for i in range(2):
        pygame.draw.rect(road_surface, ROAD, pygame.Rect(WIDTH // 2 - 125 + i, HEIGHT // 2 - 250, 250, 100))
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 + i, HEIGHT // 2 - 250, 250, 250), -math.pi / 3, math.pi / 2, 100)
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 + 90 + i, HEIGHT // 2 - 150, 250, 250), 2 * math.pi / 3, 3 * math.pi / 2, 100)
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 + i, HEIGHT // 2, 400, 400), -math.pi, math.pi / 2, 100)
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 - 150 + i, HEIGHT // 2 + 75, 250, 250), 0, math.pi / 2, 100)
        pygame.draw.rect(road_surface, ROAD, pygame.Rect(WIDTH // 2 - 125 + i, HEIGHT // 2 + 75, 100, 100))
        pygame.draw.arc(road_surface, ROAD, (WIDTH // 2 - 325 + i, HEIGHT // 2 - 250, 425, 425), math.pi / 2, 3 * math.pi / 2,100)
    return road_surface


def main():
    clock = pygame.time.Clock()

    # Create a car instance
    car = CarGame(WIDTH // 2, HEIGHT // 2 - 200)
    road_surface = draw_track()
    road_mask = pygame.mask.from_threshold(road_surface, ROAD, (1, 1, 1, 255))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Fill the screen with background color
        WIN.fill(BG)

        WIN.blit(road_surface, (0, 0))


        # Move and draw the car
        car.move()
        car.draw_timer(WIN)

        if not car.check_on_track():
            car.reset()  # Teleport car back to start if off-road

        car.update_trail()  # Update the trail
        car.draw_trail(WIN)  # Draw the trail before the car
        car.draw(WIN, road_mask)

        pygame.display.update()
        clock.tick(FPS)
        car.update_timer(clock.get_time())

# Run the main game loop
main()
