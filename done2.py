import numpy as np
import pygame
import math

WHITE_COLOR = (255, 255, 255)

class Envir:
    def __init__(self, dim):
        self.black = (0, 0, 0)
        self.brown = (128, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

        self.height = dim[0]
        self.width = dim[1]

        pygame.display.set_caption("Test")
        self.map = pygame.display.set_mode((self.width, self.height))

        self.font = pygame.font.SysFont("arial", 20)
        self.text = self.font.render("default", True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dim[1] - 500, dim[0] - 100)
        self.trail_set = []

        self.total_distance = 0  # Tổng quãng đường đã đi được

    def info(self, vx, vy, theta):
        text = f"Vx = {vx:.2f}, Vy = {vy:.2f}, Theta = {theta:.2f}"
        self.text = self.font.render(text, True, self.black, self.white)
        self.map.blit(self.text, self.textRect)

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.red, (self.trail_set[i][0], self.trail_set[i][1]),(self.trail_set[i + 1][0], self.trail_set[i + 1][1]))
        if len(self.trail_set) > 30000:  # Sửa thành len() để lấy kích thước danh sách
            self.trail_set.pop(0)
        self.trail_set.append(pos)

    def robot_frame(self, pos, rotation):
        n = 80
        centerx, centery = pos
        x_axis = (centerx + n * math.cos(rotation), centery + n * math.sin(rotation))
        y_axis = (centerx + n * math.cos(rotation + math.pi / 2), centery + n * math.sin(rotation + math.pi / 2))
        pygame.draw.line(self.map, self.blue, pos, x_axis, 3)
        pygame.draw.line(self.map, self.green, pos, y_axis, 3)

    def robot_sensor(self, pos, points):
        for point in points:
            pygame.draw.line(self.map, (120, 180, 0), pos, point)
            pygame.draw.circle(self.map, self.blue, point, 5)

class Neuron:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.rand(num_inputs, num_outputs)  # Khởi tạo ma trận trọng số ngẫu nhiên

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return 1 / (1 + np.exp(-weighted_sum))

def objective_function(params, destination):
    robot_x, robot_y = params[:2]
    dest_x, dest_y = destination
    distance = np.sqrt((robot_x - dest_x) ** 2 + (robot_y - dest_y) ** 2)
    performance = 1 / distance
    return performance

class Robot:
    def __init__(self, startpos, Img, width, num_inputs):
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0

        self.vx = 0
        self.vy = 0
        self.theta_dot = 0
    self.sensor_data = [0, 0, 0, 0, 0, 0]
    self.points = []  # Khởi tạo thuộc tính points
    self.update_sensor_data()

    self.sensor_data = [0, 0, 0, 0, 0, 0]
    self.update_sensor_data()
    self.active = True
    self.img = pygame.image.load(Img)
    self.img = pygame.image.load(Img).convert_alpha()
    self.img = pygame.transform.scale(self.img, (50, 50))

    self.rotated = self.img
    self.rect = self.rotated.get_rect(center=(self.x, self.y))

    self.visible = True  # Thêm thuộc tính để xác định liệu xe có hiển thị hay không

    self.total_distance = 0  # Tổng quãng đường đã đi được

    def draw(self, map):
        if self.visible:  # Chỉ vẽ xe nếu nó là hiển thị
            map.blit(self.rotated, self.rect)

    def track_collision_detected(self):
        for x, y in [(int(self.x), int(self.y))]:  # Chỉ kiểm tra va chạm cho robot hiện tại
            if not (0 <= x < track.get_width() and 0 <= y < track.get_height()):
                return True  # Nếu robot ra khỏi biên của track, coi như có va chạm
            if track.get_at((x, y)) == WHITE_COLOR:
                return True
        return False

    def move(self, event=None, destination=None):
        N = 10
        npar = 30
        mini = -1
        maxi = 1
        max_iteration = 40
        c1 = 0.5
        c2 = 0.5

        P = mini * np.ones((N, npar)) + (maxi - mini) * np.random.rand(N, npar)
        V = np.zeros((N, npar))
        gJbest = np.inf
        gbest = None  # Khởi tạo gbest trước khi sử dụng

        for j in range(1, max_iteration + 1):
            destination = (1090, 610)

            J = np.array([objective_function(P[i, :20], destination) for i in range(N)])
            pJbest = np.max(J)
            pbest_index = np.argmax(J)

            if pJbest < gJbest:
                gJbest = pJbest
                gbest = P[pbest_index, :]

            r1 = np.random.rand(N, npar)
            r2 = np.random.rand(N, npar)
            V = V + c1 * r1 * (P[pbest_index, :] - P) + c2 * r2 * (gbest - P)
            P = P + V

        self.vxg, self.vyg, self.theta_d = 10, 50, gbest[2]
        inv_R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                [np.sin(self.theta), np.cos(self.theta), 0],
                [0, 0, 1]])
        l = 10
        inv_J1 = np.array([[1 / np.sqrt(3), 0, -1 / np.sqrt(3)],
                         [-1 / 3, 2 / 3, -1 / 3],
                        [-1 / (3 * l), -1 / (3 * l), -1 / (3 * l)]])
        r = 3
        J2 = np.array([[r, 0, 0],
                    [0, r, 0],
                    [0, 0, r]])
        x = np.array([[self.vxg], [self.vyg], [self.theta_d]])

        V = np.linalg.inv(J2) @ np.linalg.inv(inv_J1) @ np.array([[self.vxg], [self.vyg], [self.theta_d]])

        self.v1 = V[0, 0]
        self.v2 = V[1, 0]
        self.v3 = V[2, 0]
        AAA = inv_R @ inv_J1 @ J2 @ np.array([[self.v1], [self.v2], [self.v3]])
        self.vx = AAA[0, 0]
        self.vy = AAA[1, 0]
        self.theta_dot = AAA[2, 0]

        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.theta = self.theta + self.theta_dot * dt
        self.rotated = pygame.transform.rotozoom(self.img, -math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        self.total_distance += math.sqrt(self.vx ** 2 + self.vy ** 2) * dt  # Cập nhật tổng quãng đường đã đi được

    def update_sensor_data(self):
        angles = [self.theta, np.pi / 3 + self.theta, 2 * np.pi / 3 + self.theta, np.pi + self.theta, 4 * np.pi / 3 + self.theta, 5 * np.pi / 3 + self.theta]
        edge_points = []
        edge_distances = []
        self.points = edge_points
        for angle in angles:
            distance = 0
            edge_x, edge_y = (int(self.x), int(self.y))
            while track.get_at((int(edge_x), int(edge_y))) != WHITE_COLOR:
                edge_x += int(distance * math.cos(angle))
                edge_y += int(distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.sensor_data = edge_distances

def check_active_robots(robots):
    for robot in robots:
        if robot.active:
            return True
    return False


pygame.init()
pygame.display.set_mode((1280, 720))
track = pygame.image.load('track.png').convert_alpha()
track_copy = track.copy()

start = (200, 300)
dims = (720, 1280)
running = True
dt = 0

lasttime = pygame.time.get_ticks()
enviroment = Envir(dims)

num_inputs_per_robot = 20
num_robots = 5
robots = [Robot(start, "liv.jpg", 1, num_inputs_per_robot) for _ in range(num_robots)]

# Trước khi vào vòng lặp chính, khởi tạo gbest và pbest ban đầu
N = 99999
max_iteration = 100
# Trước khi vào vòng lặp chính, khởi tạo gbest và pbest ban đầu
gbest = None
pbest = [float('-inf')] * N  # Khởi tạo pbest với giá trị âm vô cùng
destination = [1090,610]
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    for j in range(1, max_iteration + 1):
        for i, robot in enumerate(robots):
            robot.move()  # Update position and velocity of each robot
            if robot.track_collision_detected():
                robot.active = False
            robot.update_sensor_data()  # Update sensor data for each robot

            # Tính hiệu suất của particle hiện tại
            performance = objective_function((robot.x, robot.y), destination)

            # Cập nhật pbest của particle
            if performance > pbest[i]:
                pbest[i] = performance

            # Cập nhật gbest nếu có
            if gbest is None or performance > gbest:
                gbest = performance
        dt = (pygame.time.get_ticks() - lasttime) / 600
        lasttime = pygame.time.get_ticks()
        pygame.display.update()
        enviroment.map.blit(track, (0, 0))

        active_robots_exist = check_active_robots(robots)
        if not active_robots_exist:
            last_theta_d = robots[0].theta_d if len(robots) > 0 else 0  
            robots = [Robot(start, "liv.jpg", 1, num_inputs_per_robot) for _ in range(num_robots)]
            for robot in robots:
                robot.theta_d = last_theta_d  

        for robot in robots:
            if robot.active:
                robot.draw(enviroment.map)
                enviroment.robot_frame((robot.x, robot.y), robot.theta)
                enviroment.robot_sensor((robot.x, robot.y), robot.points)
                #enviroment.trail((robot.x, robot.y))
                enviroment.info(robot.vx, robot.vy, robot.theta)
                enviroment.total_distance = robot.total_distance  # Cập nhật tổng quãng đường đã đi được của môi trường

        # In ra giá trị gbest và pbest sau mỗi vòng lặp
        print("Iteration:", j)
        print("Gbest:", gbest)
        print("Pbest:", pbest)