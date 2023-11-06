from dolphin import event, controller, savestate, memory
import socket

SAVE_STATE_PATH = "/home/jflanag5/Documents/classwork/fall-2023/intro-to-ai/project/ghost_valley"

def reset_environment():
    """Reset the game environment to the initial save state."""
    savestate.load_from_file(SAVE_STATE_PATH)


def get_state():
    """Read the current game state and return percent, time, and velocity."""
    current_time = memory.read_u32(int(0x80E44744))
    current_percent = (memory.read_u32(int(0x80E44644)) - 1065328428) / (4200000)
    if current_percent < 2:
        current_percent /= 2
    else:
        current_percent -= 1
    velocity = memory.read_s32(int(0x80E4D5B4))
    return current_percent, current_time, velocity


def set_sharp_turn_right():
    """Move car in sharp right direction."""
    controller.set_wii_nunchuk_buttons(0, {"StickX": 1.0})
    controller.set_wiimote_buttons(0, {"A": True, "B": True})


def set_slight_turn_right():
    """Move car in slight right direction."""
    controller.set_wii_nunchuk_buttons(0, {"StickX": 0.5})
    controller.set_wiimote_buttons(0, {"A": True, "B": True})


def set_straight():
    """Move car in forwards direction."""
    controller.set_wii_nunchuk_buttons(0, {"StickX": 0})
    controller.set_wiimote_buttons(0, {"A": True, "B": False})


def set_slight_turn_left():
    """Move car in slight left direction."""
    controller.set_wii_nunchuk_buttons(0, {"StickX": -0.5})
    controller.set_wiimote_buttons(0, {"A": True, "B": True})


def set_sharp_turn_left():
    """Move car in sharp left direction."""
    controller.set_wii_nunchuk_buttons(0, {"StickX": -1.0})
    controller.set_wiimote_buttons(0, {"A": True, "B": True})


# Set up the server
HOST = '127.0.0.1'
PORT = 65431
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
print('Listening for commands...')
conn, addr = s.accept()
print('Connected by', addr)


def handle_socket_commands():
    """Handles and executes socket commands."""
    controller.set_wiimote_buttons(0, {"A": True})
    data = conn.recv(1024)
    if not data:
        return

    commands = data.decode('utf-8').split('\n')
    for command in commands:
        if not command:
            continue
        if command == 'reset':
            reset_environment()
        elif command == 'get-state':
            state = get_state()
            conn.sendall(str(state).encode('utf-8'))
        elif command == 'move-sharp-left':
            set_sharp_turn_left()
        elif command == 'move-slight-left':
            set_slight_turn_left()
        elif command == 'move-straight':
            set_straight()
        elif command == 'move-slight-right':
            set_slight_turn_right()
        elif command == 'move-sharp-right':
            set_sharp_turn_right()


event.on_frameadvance(handle_socket_commands)
