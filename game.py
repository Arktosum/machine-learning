import pygame
import sys




class Game:
    def __init__(self,drawFunction):
        # Initialize Pygame
        pygame.init()

        # Set the dimensions of the window
        self.WIDTH, self.HEIGHT = 800, 600
        self.drawFunction  = drawFunction
        # Set colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("2D Plot in Pygame")

    # Function to draw axes
    def draw_axes(self):
        pygame.draw.line(self.screen, self.BLACK, (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT), 2)  # X-axis
        pygame.draw.line(self.screen, self.BLACK, (0, self.HEIGHT // 2), (self.WIDTH, self.HEIGHT // 2), 2)  # Y-axis

    # Function to plot data points
    def plot_points(self,data):
        for point in data:
            x, y = point
            pygame.draw.circle(self.screen, self.RED, (self.WIDTH // 2 + int(x), self.HEIGHT // 2 - int(y)), 3)

    # Function to draw grid lines
    def draw_grid(self):
        # Horizontal lines
        for y in range(0, self.HEIGHT // 2, 50):
            pygame.draw.line(self.screen, self.GRAY, (0, self.HEIGHT // 2 - y), (self.WIDTH, self.HEIGHT // 2 - y), 1)
            pygame.draw.line(self.screen, self.GRAY, (0, self.HEIGHT // 2 + y), (self.WIDTH, self.HEIGHT // 2 + y), 1)
        # Vertical lines
        for x in range(0, self.WIDTH // 2, 50):
            pygame.draw.line(self.screen, self.GRAY, (self.WIDTH // 2 - x, 0), (self.WIDTH // 2 - x, self.HEIGHT), 1)
            pygame.draw.line(self.screen, self.GRAY, (self.WIDTH // 2 + x, 0), (self.WIDTH // 2 + x, self.HEIGHT), 1)
    def run(self):
        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(self.WHITE)  # Fill the screen with white

            self.draw_axes()  # Draw axes
            self.draw_grid()  # Draw grid lines
            # Example data points (replace this with your own data)
            self.drawFunction()
            pygame.display.flip()  # Update the display

        pygame.quit()
        sys.exit()