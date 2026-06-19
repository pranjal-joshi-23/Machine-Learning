import socket

host = '0.0.0.0'
port = 65433

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"Server listening on {host}:{port}")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

while True:
    message = input("Enter message to send: ")
    conn.sendall(message.encode())

conn.close()
server_socket.close()