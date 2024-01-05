from Trash_Main import UI
from Trash_Main import ser_com
from multiprocessing import Process
from multiprocessing import shared_memory

if __name__ == '__main__':
    # 使用两组共享内存，实现两独立的进程之间的参数传递
    try:
        shm_send=shared_memory.SharedMemory(name='ser_send')
        shm_send.close()
        shm_send.unlink()
        shm_send=shared_memory.SharedMemory(name='ser_send',create=True, size=100)
    except FileNotFoundError:
        shm_send=shared_memory.SharedMemory(name='ser_send',create=True, size=100)
        
    try:
        shm_receive=shared_memory.SharedMemory(name='ser_receive')
        shm_receive.close()
        shm_receive.unlink()
        shm_receive=shared_memory.SharedMemory(name='ser_receive',create=True, size=100)
    except FileNotFoundError:
        shm_receive=shared_memory.SharedMemory(name='ser_receive',create=True, size=100)

    buf_send=shm_send.buf
    buf_send[:4]=bytearray([0,0,0,0])
    buf_receive=shm_receive.buf
    buf_receive[:5]=bytearray([5,0,0,0,0])

    sc = Process(target = ser_com)
    ui = Process(target = UI)
    # while True:
    #     try:
    #         sc.start()
    #         ui.start()

    #         sc.join()
    #         ui.join()
    #     except Exception as e:
    #         print("reboot serial com......")
    
    sc.start()
    ui.start()

    sc.join()
    ui.join()

    shm_send.unlink()
    shm_receive.unlink()
