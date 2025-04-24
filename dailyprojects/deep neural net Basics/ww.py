import time
import os

def clear_screen():
  """Clears the console screen."""
  os.system('cls' if os.name == 'nt' else 'clear')

def draw_ball(position):
  """Draws a ball at the given vertical position."""
  for i in range(10):
    if i == position:
      print(" " * 5 + "*")
    else:
      print(" " * 5 + " ")

def animate_ball():
  """Animates a ball moving up and down."""
  position = 0
  direction = 1  # 1 for down, -1 for up

  for _ in range(30):
    clear_screen()
    draw_ball(position)
    time.sleep(0.1)
    position += direction
    if position == 9:
      direction = -1
    elif position == 0:
      direction = 1

def draw_wave(frame):
  """Draws a simple wave pattern."""
  amplitude = 3
  length = 15
  for i in range(amplitude * 2 + 1):
    line = ""
    for j in range(length):
      if i == amplitude + int(amplitude * (frame % (length * 2) - length) / length):
        line += "*"
      else:
        line += " "
    print(line)

def animate_wave():
  """Animates a simple wave moving across the screen."""
  for frame in range(60):
    clear_screen()
    draw_wave(frame)
    time.sleep(0.05)

def draw_blinking_star(frame):
  """Draws a blinking star."""
  if frame % 5 == 0:
    print("    *")
  else:
    print("     ")

def animate_blinking_star():
  """Animates a blinking star."""
  for frame in range(20):
    clear_screen()
    draw_blinking_star(frame)
    time.sleep(0.2)

# Example usage:
print("Animating a bouncing ball:")
animate_ball()

print("\nAnimating a wave:")
animate_wave()

print("\nAnimating a blinking star:")
animate_blinking_star()