from dolphin import event, memory, savestate

SAVE_STATE_PATH = "/home/jflanag5/Documents/classwork/fall-2023/intro-to-ai/project/ghost_valley"

savestate.load_from_file(SAVE_STATE_PATH)
frame_counter = 0

while True:
    await event.frameadvance()
    frame_counter += 1

    if frame_counter % 30 == 0:
        current_lap = memory.read_u8(int(0x80E4465D))
        current_time = memory.read_u32(int(0x80E44744))
        current_percent = (memory.read_u32(int(0x80E44644)) - 1065328428)/(4200000)
        if current_percent < 2:
            current_percent /= 2
        else:
            current_percent -= 1
        current_velocity = memory.read_s32(int(0x80E4D5B4))


        print(f"current lap: {current_lap}")
        print(f"current time: {current_time}")
        print(f"current progress: {current_percent}")
        print(f"current velocity: {current_velocity}")
