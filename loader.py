import time

def load_animation():
    symbols = [
        '⣾', '⣷', '⣯', '⣟', '⡿',
        '⢿', '⣻', '⣽'
    ]
    while True:
        for symbol in symbols:
            print(f'\r{symbol}', end='', flush=True)
            time.sleep(0.1)

if __name__ == "__main__":
    load_animation()
