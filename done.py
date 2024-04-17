import pygame
import math
import numpy as np
import pandas as pd
from time import sleep
from scipy.special import expit

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

class Envir:
    def __init__(self, dim):
        
        self.height = dim[0]
        self.width = dim[1]
        
        self.edge_distances = [0,0,0,0,0,0,0,0]
        
        pygame.display.set_caption('Cooking ROBOT')
        self.map = pygame.display.set_mode((self.width, self.height))
        
        self.font = pygame.font.SysFont('arial', 30)
        self.text = self.font.render('default', True, black, white)
        self.textRect = self.text.get_rect()
        self.textRect2 = self.text.get_rect()

        self.textRect.center = (dim[1] - 1100, dim[0] - 80)
        self.textRect2.center = (dim[1] - 1100, dim[0] - 40)
        # self.textRect.center = (dim[1] - 830, dim[0] - 40)
        # self.textRect2.center = (dim[1] - 1600, dim[0] - 40)
        
    def sensor_info(self, sensor_data):
        text  = f"sensor: {sensor_data}"
        self.text = self.font.render(text, True, black, white)
        self.map.blit(self.text, self.textRect)
    
    def info(self, vx, vy, theta, v1, v2, v3):
        text = f'Vx = {np.round(vx, 2)}, Vy = {np.round(vy, 2)}, Theta = {np.round(theta, 2)}, V1 = {np.round(v1, 2)}, V2 = {np.round(v2, 2)}, V3 = {np.round(v3, 2)}'
        self.text = self.font.render(text, True, black, white)
        self.map.blit(self.text, self.textRect)
        
    def robot_frame(self, pos, rotation):
        n = 80
        centerx, centery = pos
        x_axis = (centerx + n*math.cos(rotation), centery + n*math.sin(rotation))
        y_axis = (centerx + n*math.cos(rotation + math.pi/2), centery + n*math.sin(rotation + math.pi/2))
        pygame.draw.line(self.map, blue, pos, x_axis, 4)
        pygame.draw.line(self.map, red, pos, y_axis, 4)
    
    def robot_sensor(self, pos, points):
        for point in points:
            pygame.draw.line(self.map, (0,255,0), pos, point)
            pygame.draw.circle(self.map, (0, 255, 0), point, 5)

    def best_fitness(self,ite,len,lap,pJbest):
        text = f'Iteration: {ite}, Robot: {len}, Lap: {lap}, pJbest= {np.round(pJbest, 2)}, gJbest= {np.round(gJbest, 2)}'
        self.text = self.font.render(text, True, black, white)
        self.map.blit(self.text, self.textRect)
        
        
class Robot:
    def __init__(self, startpos, Img, width):
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = np.pi/2
        self.v1 = 0
        self.v2 = 0
        self.v3 = 0
        self.vx = 0
        self.vy = 0
        self.theta_dot = 0
        self.vxg = 0
        self.vyg = 0
        self.theta_d = 0
        self.sensor_data = [0,0,0,0,0,0]
        self.points = []
        self.xy_stored = []
        self.crash = False
        self.finish = False
        
        self.P=[]
        self.y_out=[]
        self.ee = X
        self.V=[]
        self.W=[]
        self.cost_fnc = 0
        self.lap=0
        
        self.img = pygame.image.load(Img)
        self.img = pygame.transform.scale(self.img, (50,50))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center = (self.x, self.y))
        
    def update_sensor_data(self):
        angles = [self.theta, np.pi/3 + self.theta, 2*np.pi/3 + self.theta,
                  np.pi + self.theta, 4*np.pi/3 + self.theta, 5*np.pi/3 + self.theta]
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = (int(self.x), int(self.y))
            while track_copy.get_at((edge_x, edge_y)) != white:
                edge_x = int(self.x + distance * math.cos(angle))
                edge_y = int(self.y + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.sensor_data = edge_distances
        self.points = edge_points
        
    def draw(self, map):
        map.blit(self.rotated, self.rect)
    
    def trail(self):
        for i in range(0, len(self.xy_stored) - 1):
            pygame.draw.line(environment.map,red,(self.xy_stored[i][0],self.xy_stored[i][1]),(self.xy_stored[i+1][0],self.xy_stored[i+1][1]))
        if self.xy_stored.__sizeof__() > 30000:
            self.xy_stored.pop(0)
        self.xy_stored.append((robot.x, robot.y))

    def Neuron_fnc(self,lamda):
        abc=[]
        for j in self.sensor_data:
            abc.append(j*0.01)
        net_h = np.dot(self.V.T, np.hstack((self.ee[0]*0.01,self.ee[1]*0.01,self.ee[2],abc)))
        # y_h = expit(net_h)
        y_h = 2 / (1 + np.exp(-lamda * net_h)) - 1
        net_0 = np.dot(self.W.T, y_h)
        self.theta_d=net_0[0]
        # self.v1=net_0[0]
        # self.v2=net_0[1]
        # self.v3=net_0[2]
    
    def forward_fcn(self):
        r = 3
        l = 10
        inv_R = np.array([  [np.cos(self.theta), -np.sin(self.theta), 0],
                            [np.sin(self.theta), np.cos(self.theta), 0],
                            [0, 0, 1]])
        inv_j1 = np.array([[1/np.sqrt(3), 0, -1/np.sqrt(3)], [-1/3, 2/3, -1/3], [-1/(3*l), -1/(3*l), -1/(3*l)]])
        j2 = np.array([[r, 0, 0], [0, r, 0], [0, 0, r]])

        V = np.linalg.inv(j2) @ np.linalg.inv(inv_j1) @ np.array([[self.vxg], [self.vyg], [self.theta_d]])
        self.v1 = V[0, 0]
        self.v2 = V[1, 0]
        self.v3 = V[2, 0]

        AAA = inv_R @ inv_j1 @ j2 @ np.array([[self.v1], [self.v2], [self.v3]])
        self.vx = AAA[0, 0]
        self.vy = AAA[1, 0]
        self.theta_dot = AAA[2, 0]
        self.x = self.x + self.vx*dt
        self.y = self.y + self.vy*dt
        self.theta = self.theta + self.theta_dot*dt
        self.rotated = pygame.transform.rotozoom(self.img, -math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center = (self.x, self.y))
        self.y_out = np.array([self.x,self.y,self.theta])
        self.ee = X - self.y_out

    def Cost_fnc(self):
        self.cost_fnc =0

    def check_crash(self):
        edge_x, edge_y = (int(self.x), int(self.y))
        if track_copy.get_at((edge_x, edge_y)) == white:
            self.crash = True
        elif track_copy.get_at((edge_x, edge_y)) == green:
            self.finish = True


pygame.init()
pygame.display.set_mode((1280, 720))
track = pygame.image.load("track.png")
track_copy = track.copy()

start=(200,300)
end_point=(1090,710)
dims = (720, 1280)
running = True
dt = 0
lasttime = pygame.time.get_ticks()
environment = Envir(dims)


N = 50  # cá thể
la = 20  # noron lớp ẩn
dv = 9 # ngõ vào noron
dr = 1 # ngõ ra noron
npar = dv*la + la*dr # 30 tham số
min_val = -1
max_val = 1
max_iteration = 1000
c1 = 0.2
c2 = 0.2

# Khởi tạo vị trí và vận tốc ngẫu nhiên
P = np.random.uniform(min_val, max_val, size=(N, npar))
P_save = []
V1 = np.zeros((N, npar))

X = np.array([end_point[0],end_point[1],np.pi/2])
gJbest = 0
gbest=[]


for i in range(max_iteration):
    if running:
        robots=[]

        for iii in range(N):
            robot = Robot(start, "liv.jpg", 1)
            robot.theta_d = np.pi/2
            robot.P = P[iii,:]
            robot.V = P[iii,0:dv*la].reshape(dv,la)
            robot.W = P[iii,dv*la:(dv+dr)*la].reshape(la,dr)
            robots.append(robot)
        pJbest = 0
        pbest = []
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for robot in robots:
                if robot.crash or robot.finish or robot.lap >= 2000:
                    # lưu fitness
                    if robot.cost_fnc > pJbest:
                        pJbest = robot.cost_fnc
                        pbest = robot.P
                    if robot.cost_fnc > gJbest:
                        gJbest = robot.cost_fnc
                        gbest = robot.P
                    # loại robot
                    robots.remove(robot)

                robot.vxg=100

                robot.Neuron_fnc(1)
                robot.forward_fcn()
                

                robot.check_crash()
                robot.lap+=1
                
                robot.update_sensor_data()
                robot.trail()
                robot.draw(environment.map)
                environment.robot_frame((robot.x, robot.y), robot.theta)
                environment.robot_sensor((robot.x, robot.y), robot.points)
                
            if len(robots)==0:
                break
            
            dt = (pygame.time.get_ticks() - lasttime)/1000
            lasttime = pygame.time.get_ticks()
            pygame.display.update()
            environment.map.blit(track, (0, 0))
            # environment.info(robot.vx, robot.vy, robot.theta, robot.v1, robot.v2, robot.v3)
            # environment.sensor_info(robot.sensor_data)
            environment.best_fitness(i+1,len(robots),robot.lap,pJbest)
        # Cập nhật sau mỗi iteration
        V1 = V1 + c1 * np.random.rand() * (pbest - P) + c2 * np.random.rand() * (gbest - P)
        P = P + V1
            

        
    