def format_time(t):
    ms = int((t - int(t))*100)
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return f'{h:d}:{m:02d}:{s:02d}.{ms:03d}'
    else:
        return f'{m:02d}:{s:02d}.{ms:03d}'