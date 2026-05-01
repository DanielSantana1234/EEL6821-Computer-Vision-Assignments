import pygame
import numpy as np

WIDTH, HEIGHT = 800, 600
FPS = 60
BG_COLOR = (0, 0, 0)
LADDER_COLOR = (255, 255, 255)
GROUND_COLOR = (40, 40, 40)

def make_background(width: int, height: int) -> np.ndarray:
    bg = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        shade = int(15 + 70 * (y / max(1, height - 1)))
        bg[y, :] = shade
    return bg


def make_ladder_mask(width: int = 220, height: int = 220, spacing: int = 24) -> np.ndarray:
    ladder = np.zeros((height, width), dtype=np.uint8)

    left_rail = width // 3
    right_rail = width - left_rail

    ladder[12:height - 12, left_rail - 2:left_rail + 2] = 255
    ladder[12:height - 12, right_rail - 2:right_rail + 2] = 255

    for y in range(16, height - 16, spacing):
        ladder[y - 1:y + 2, left_rail:right_rail] = 255

    return ladder


def warp_perspective(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    height1, width1 = img1.shape
    height2, width2 = img2.shape

    projection_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -1 / 500, 1]
    ], dtype=float)

    k1 = width1 * 0.5
    k2 = height1 * 0.70
    translation_matrix = np.array([
        [1, 0, 0, k1],
        [0, 1, 0, k2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    output = img1.copy()

    for x in range(width2):
        for y in range(height2):
            curr_coord = img2[y, x]
            if curr_coord == 0:
                continue

            z = (y - height2 / 2.0) * 4.0

            pixel_coord = np.array([x - width2 / 2.0, y - height2 / 2.0, z, 1], dtype=float)
            result = np.dot(projection_matrix, pixel_coord)
            result_translated = np.dot(translation_matrix, result)

            if abs(result_translated[3]) < 1e-8:
                continue

            result_translated[0] /= result_translated[3]
            result_translated[1] /= result_translated[3]

            x_new = int(round(result_translated[0]))
            y_new = int(round(result_translated[1]))

            if 0 <= x_new < width1 and 0 <= y_new < height1:
                output[y_new, x_new] = curr_coord

    return output


def to_pygame_surface(gray_img: np.ndarray) -> pygame.Surface:
    rgb = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Perspective Warp Ladder (Pygame)")
    clock = pygame.time.Clock()

    gray_img1 = make_background(WIDTH, HEIGHT)
    image_2 = make_ladder_mask()
    warped = warp_perspective(gray_img1, image_2)
    warped_surface = to_pygame_surface(warped)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.blit(warped_surface, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()