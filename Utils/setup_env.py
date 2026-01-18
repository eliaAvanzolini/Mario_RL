import pygame
import imageio
import os

CUSTOM_REWARDS = {
    "time": -0.1,  # per second that passes by
    "death": -100.,  # mario dies
    "extra_life": 100.,  # mario gets an extra life, which includes getting 100th coin
    "mushroom": 20.,  # mario eats a mushroom to become big
    "flower": 25.,  # mario eats a flower
    "mushroom_hit": -10.,  # mario gets hit while big (super mario)
    "flower_hit": -15.,  # mario gets hit while fire mario
    "coin": 15.,  # mario gets a coin
    "score": 15.,  # mario hit enemies
    "victory": 1000  # mario win
}


def init_pygame():
    """Initializes pygame display for visualization."""
    pygame.init()
    screen = pygame.display.set_mode((240, 256))
    pygame.display.set_caption("Super Mario Bros")
    return screen


def custom_rewards(name, tmp_info):
    """
    Calculates a custom reward based on changes in game state between frames.

    Args:
        name (dict): Current info dictionary from the environment step.
        tmp_info (dict): Info dictionary from the previous step.

    Returns:
        tuple: (reward, name) - the calculated custom reward and the updated info dictionary.
    """
    reward = 0

    # detect score change (e.g., hitting an enemy/block)
    if tmp_info['score'] != name['score']:
        reward += CUSTOM_REWARDS['score']

    # detect x_pos change (encourage forward movement)
    if tmp_info['x_pos'] != name['x_pos']:
        reward += name['x_pos'] - tmp_info['x_pos']

    # detect time change (penalize running out of time)
    if tmp_info['time'] != name['time']:
        reward += CUSTOM_REWARDS['time']

    # detect if finished (victory)
    if name['x_pos'] > 3159 or (tmp_info['flag_get'] != name['flag_get'] and name['flag_get']):
        print('Victory\n')
        reward += CUSTOM_REWARDS['victory']

    # detect deaths
    # 'TimeLimit.truncated' indicates a death or time-out in the gym-super-mario-bros environment
    if 'TimeLimit.truncated' in name and name['x_pos'] < 3159 and name['life'] < tmp_info['life']:
        reward += CUSTOM_REWARDS["death"]

    # detect extra lives (and coin overflow at 100 coins)
    if tmp_info['life'] != name['life'] and name["life"] > 2:
        reward += CUSTOM_REWARDS['extra_life']

    # detect getting a coin
    if tmp_info['coins'] != name['coins']:
        reward += CUSTOM_REWARDS['coin']
        # Give a bigger bonus for collecting more coins (optional, based on original code)
        if name["coins"] > 6:
            reward += 500

    # detect power-up changes (mushroom, flower, or getting hit without dying)
    if tmp_info['status'] != name['status']:
        
        # Current status: 'small', 'tall' (super mario), 'fireball' (fire mario)

        # Got hit while Super Mario (tall) -> Small Mario (small)
        if tmp_info['status'] == 'tall' and name['status'] == 'small':
            reward += CUSTOM_REWARDS['mushroom_hit']

        # Got hit while Fire Mario (fireball) -> Super Mario (tall)
        elif tmp_info['status'] == 'fireball' and name['status'] == 'tall':
            reward += CUSTOM_REWARDS['flower_hit']

        # Ate a Flower: becomes Fire Mario
        elif name['status'] == 'fireball':
            reward += CUSTOM_REWARDS['flower']

        # Ate a Mushroom: becomes Super Mario (only if it wasn't a flower hit)
        elif name['status'] == 'tall':
            reward += CUSTOM_REWARDS['mushroom']

    return reward, name


def show_state(enviroment, ep=0, info=""):
    """Displays the current game state using pygame."""
    screen = pygame.display.get_surface()
    image = enviroment.render(mode='rgb_array')
    image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    screen.blit(image, (0, 0))
    pygame.display.flip()
    pygame.display.set_caption(f"Episode: {ep} {info}")
    pygame.time.delay(50)  # Add a delay to slow down the visualization


def generate_gif(image_folder, output_gif, file_extension='.png', fps=60):
    """Generates a GIF from a folder of images."""
    images = []

    # Get only files with the specified extension in the image folder
    for file in os.listdir(image_folder):
        if file.endswith(file_extension):
            img_path = os.path.join(image_folder, file)
            images.append(imageio.imread(img_path))

    imageio.mimsave(output_gif, images, fps=fps)


def generate_images_mario(enviroment, ep, state):
    """Saves the current frame as an image."""
    image = enviroment.render(mode='rgb_array')
    # Save the image with a unique name based on the episode and state
    imageio.imwrite(f"mario_image_episode_{ep}_{state}.png", image)