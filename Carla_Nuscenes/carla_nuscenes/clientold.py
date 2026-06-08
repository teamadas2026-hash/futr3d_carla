import carla
from .sensor import *
from .vehicle import Vehicle
from .walker import Walker
import math
from .utils import generate_token,get_nuscenes_rt,get_intrinsic,transform_timestamp,clamp
import random
import socket
def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1

class Client:
    def __init__(self,client_config,random_seed=0):
        random.seed(random_seed)
        self.random_seed=random_seed
        self.client = carla.Client(client_config["host"],client_config["port"])
        self.client.set_timeout(client_config["time_out"])
        
        
        
    def clear_all_vehicles(self,world):
        """遍历销毁所有车辆"""
        actors = world.get_actors()
        # vehicles = actors.filter('vehicle.*')
        vehicles = actors.filter('vehicle.*')

        for vehicle in vehicles:
            vehicle.destroy()

        print(f"已清除 {len(vehicles)} 辆车辆")
    def generate_world(self,world_config):
        print("generate world start!")
        self.client.load_world(world_config["map_name"])
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None
        self.clear_all_vehicles(self.world)
        # In client.py generate_world(), replace get_category lambda:
        def get_category(bp):
            bp_id = bp.id.lower()
            if bp_id.startswith("walker"):
                return "human.pedestrian.adult"
            if any(t in bp_id for t in ["motorcycle", "bike", "vespa", "yamaha", "harley", "kawasaki", "scooter"]):
                return "vehicle.motorcycle"
            if any(t in bp_id for t in ["truck", "carlacola", "firetruck"]):
                return "vehicle.truck"
            if any(t in bp_id for t in ["ambulance"]):
                return "vehicle.truck"  # closest nuScenes class
            if any(t in bp_id for t in ["bus"]):
                return "vehicle.bus.rigid"
            if any(t in bp_id for t in ["van", "sprinter", "transit"]):
                return "vehicle.car"    # nuScenes has no van class
            if bp_id.startswith("vehicle"):
                return "vehicle.car"
            return None

        self.category_dict = {bp.id: get_category(bp) for bp in self.world.get_blueprint_library()}
        #get_category = lambda bp: "vehicle.car" if bp.id.split(".")[0] == "vehicle" else "human.pedestrian.adult" if bp.id.split(".")[0] == "walker" else None
        #self.category_dict = {bp.id: get_category(bp) for bp in self.world.get_blueprint_library()}
        get_attribute = lambda bp: ["vehicle.moving"] if bp.id.split(".")[0] == "vehicle" else ["pedestrian.moving"] if bp.id.split(".")[0] == "walker" else None
        self.attribute_dict = {bp.id: get_attribute(bp) for bp in self.world.get_blueprint_library()}
        
        attempts = 0
        num_max_restarts = 40
        while attempts < num_max_restarts:
            try:
                trafficmanager_port = find_free_port(30000)
                print(f"Trying to initialize traffic manager on port {trafficmanager_port}", flush=True)
                self.trafficmanager = self.client.get_trafficmanager(trafficmanager_port)
                self.trafficmanager.set_synchronous_mode(True)
                self.trafficmanager.set_respawn_dormant_vehicles(True)
                print(f"traffic_manager init success, try_time={attempts}", flush=True)
                break
            except Exception as e:
                print(f"traffic_manager init fail, try_time={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                import time
                time.sleep(5)
        
        self.settings = carla.WorldSettings(**world_config["settings"])
        # 始终保持注释
        ## self.settings.substepping = True
        ## self.settings.max_substep_delta_time = 0.03125
        ## self.settings.max_substeps = 16
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
        self.world.set_pedestrians_cross_factor(1)
        self.spawn_points=self.world.get_map().get_spawn_points()
        random.shuffle(self.spawn_points)
        self.count=0

        print("generate world success!")

    def generate_scene(self,scene_config):
        print("generate scene start!")
        if scene_config["custom"]:
            self.generate_custom_scene(scene_config)
        else:
            self.generate_random_scene(scene_config)
        print("generate scene success!")

    def _get_traffic_config(self,scene_config):
        traffic = scene_config.get("traffic")
        if traffic is None:
            return None
        return {
            "cars": int(traffic.get("cars", 25)),
            "trucks": int(traffic.get("trucks", 6)),
            "bikes": int(traffic.get("bikes", 6)),
            "vans": int(traffic.get("vans", 6)),
            "walkers": int(traffic.get("walkers", 20)),
        }

    def _split_vehicle_blueprints(self):
        blueprints = list(self.world.get_blueprint_library().filter("vehicle.*"))

        def has_any_token(bp_id,tokens):
            bp_id = bp_id.lower()
            return any(token in bp_id for token in tokens)

        bike_tokens = ["motorcycle", "bike", "vespa", "yamaha", "harley", "kawasaki", "scooter"]
        truck_tokens = ["truck", "hgv", "carlacola", "firetruck", "ambulance", "bus"]
        van_tokens = ["van", "sprinter", "transit"]

        cars = []
        trucks = []
        bikes = []
        vans = []

        for bp in blueprints:
            wheels = None
            if bp.has_attribute("number_of_wheels"):
                try:
                    wheels = int(bp.get_attribute("number_of_wheels"))
                except Exception:
                    wheels = None

            if wheels == 2 or has_any_token(bp.id, bike_tokens):
                bikes.append(bp)
            elif has_any_token(bp.id, truck_tokens):
                trucks.append(bp)
            elif has_any_token(bp.id, van_tokens):
                vans.append(bp)
            else:
                cars.append(bp)

        return {
            "cars": cars,
            "trucks": trucks or cars,
            "bikes": bikes or cars,
            "vans": vans or cars,
        }

    def _spawn_vehicle_from_transform(self,bp,transform):
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            return None

        location = {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
        }
        rotation = {
            "yaw": transform.rotation.yaw,
            "pitch": transform.rotation.pitch,
            "roll": transform.rotation.roll,
        }
        vehicle = Vehicle(world=self.world,bp_name=bp.id,location=location,rotation=rotation)
        vehicle.set_actor(actor.id)
        actor.set_autopilot(True, self.trafficmanager.get_port())
        self.trafficmanager.keep_right_rule_percentage(actor, 100)
        self.trafficmanager.ignore_lights_percentage(actor, 100)
        self.trafficmanager.ignore_signs_percentage(actor, 100)
        self.trafficmanager.distance_to_leading_vehicle(actor, 5)
        self.trafficmanager.vehicle_percentage_speed_difference(actor, -20)
        self.trafficmanager.auto_lane_change(actor, True)
        self.trafficmanager.ignore_vehicles_percentage(actor, 0)
        return vehicle

    def _spawn_traffic_vehicles(self,traffic_config,ego_location):
        min_distance = 8.0
        spawn_points = list(self.spawn_points)
        random.shuffle(spawn_points)
        if ego_location is not None:
            ego_point = carla.Location(**ego_location)
            spawn_points = [p for p in spawn_points if p.location.distance(ego_point) >= min_distance]

        blueprints = self._split_vehicle_blueprints()
        vehicles = []
        for group_name in ["cars", "trucks", "bikes", "vans"]:
            count = max(0, int(traffic_config.get(group_name, 0)))
            bps = blueprints[group_name]
            attempts = 0
            max_attempts = max(10, count * 5)
            while len([v for v in vehicles if v.bp_name in [bp.id for bp in bps]]) < count and spawn_points and attempts < max_attempts:
                attempts += 1
                transform = spawn_points.pop()
                bp = random.choice(bps)
                vehicle = self._spawn_vehicle_from_transform(bp, transform)
                if vehicle is not None:
                    vehicles.append(vehicle)

        return vehicles

    def _spawn_configured_vehicles(self,vehicle_configs):
        vehicles = []
        for vehicle_config in vehicle_configs:
            bp_name = vehicle_config["bp_name"]
            transform = carla.Transform(
                carla.Location(**vehicle_config["location"]),
                carla.Rotation(**vehicle_config["rotation"])
            )
            bp = self.world.get_blueprint_library().find(bp_name)
            vehicle = self._spawn_vehicle_from_transform(bp, transform)
            if vehicle is not None:
                vehicles.append(vehicle)
        return vehicles

    def _spawn_walkers(self,count,ego_location=None):
        walkers = []
        if count <= 0:
            return walkers

        walker_bps = list(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        if not walker_bps:
            return walkers

        ego_point = carla.Location(**ego_location) if ego_location is not None else None

        for _ in range(count):
            spawn_location = self.world.get_random_location_from_navigation()
            destination = self.world.get_random_location_from_navigation()
            if spawn_location is None or destination is None:
                continue
            if ego_point is not None and spawn_location.distance(ego_point) < 6.0:
                continue
            rotation = carla.Rotation(yaw=random.uniform(0, 360))
            transform = carla.Transform(spawn_location, rotation)
            bp = random.choice(walker_bps)
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                continue
            walker = Walker(
                world=self.world,
                location={"x": spawn_location.x, "y": spawn_location.y, "z": spawn_location.z},
                rotation={"yaw": rotation.yaw, "pitch": rotation.pitch, "roll": rotation.roll},
                destination={"x": destination.x, "y": destination.y, "z": destination.z},
                bp_name=bp.id,
            )
            walker.set_actor(actor.id)
            walkers.append(walker)

        walker_controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        for walker in walkers:
            controller = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker.get_actor())
            if controller is None:
                continue
            walker.set_controller(controller.id)
            walker.start()
            controller.set_max_speed(random.uniform(1.0, 2.0))

        self.world.tick()
        return walkers

    def _spawn_configured_walkers(self,walker_configs):
        walkers = []
        for walker_config in walker_configs:
            transform = carla.Transform(
                carla.Location(**walker_config["location"]),
                carla.Rotation(**walker_config["rotation"])
            )
            bp = self.world.get_blueprint_library().find(walker_config["bp_name"])
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                continue
            walker = Walker(world=self.world,**walker_config)
            walker.set_actor(actor.id)
            walkers.append(walker)

        if not walkers:
            return walkers

        walker_controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        for walker in walkers:
            controller = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker.get_actor())
            if controller is None:
                continue
            walker.set_controller(controller.id)
            walker.start()
            controller.set_max_speed(random.uniform(1.0, 2.0))

        self.world.tick()
        return walkers

    def generate_custom_scene_different_pose(self,scene_config):
        
        if scene_config["weather_mode"] == "custom":
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])
        
        self.world.set_weather(self.weather)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])
        ego_bp_name=scene_config["ego_vehicle"]["bp_name"]
        spawn_points_number= self.count % len(self.spawn_points)
        ego_location={attr:getattr(self.spawn_points[spawn_points_number].location,attr) for attr in ["x","y","z"]}
        ego_rotation={attr:getattr(self.spawn_points[spawn_points_number].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.yaw =ego_rotation["yaw"]
        print(f"yaw:{self.yaw}")

        self.count+=1 
        # ego_location["z"]=0.015

        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        # self.ego_vehicle.get_actor().set_autopilot()
        #设计了对自车的驾驶策略
        #fuck self.trafficmanager.set_random_device_seed(self.random_seed)
        #fuck self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        #fuck self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        #fuck self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),0)
        # #fuck self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0)
        #fuck self.trafficmanager.keep_right_rule_percentage(self.ego_vehicle.get_actor(),100)
        #fuck self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),5)
        #fuck self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        #fuck self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)

        # self.vehicles = [Vehicle(world=self.world,**vehicle_config) for vehicle_config in scene_config["vehicles"]]

        reGenerate=True
        
        while reGenerate:
            
            reGenerate=False
            self.vehicles = []
            if scene_config["vehicles"] is not None:
                for vehicle_config in scene_config["vehicles"]:
                    bp_name = vehicle_config["bp_name"]
                    #在ego_vehicle随机选择一个位置
                    distance= random.uniform(9,15)
                    distance= 40
                    print(ego_rotation["yaw"])
                    
                    #角度转换至世界坐标系
                    angle = random.uniform(0, 2*math.pi)  # 0到2π的均匀分布角度
                    angle = 5/3.0*math.pi # 0到2π的均匀分布角度

                    vehicle_config["location"]["x"]=ego_location["x"]+distance*math.cos(math.radians(ego_rotation["yaw"])+angle)
                    vehicle_config["location"]["y"]=ego_location["y"]+distance*math.sin(math.radians(ego_rotation["yaw"])+angle)
                    vehicle_config["location"]["z"]=ego_location["z"]

                    
                    # for attr in ["x","y"]:
                    #     vehicle_config["location"][attr]=ego_location[attr]+random.uniform(-10,10)
                    # vehicle_config["location"]["z"]=ego_location["z"]

                    vehicle_config["rotation"]["yaw"] = random.uniform(0, 360)
                    print(vehicle_config["location"])
                    vehicle_config["rotation"]["pitch"]=ego_rotation["pitch"]
                    vehicle_config["rotation"]["roll"]=ego_rotation["roll"]
                    vehicle=Vehicle(world=self.world,bp_name=bp_name,location=vehicle_config["location"],rotation=vehicle_config["rotation"])
                    
                    self.vehicles.append(Vehicle(world=self.world,bp_name=bp_name,location=vehicle_config["location"],rotation=vehicle_config["rotation"]))
            else:
                #重复random
                nothing=random.uniform(5,10)
                nothing=random.uniform(-180, 180)
                nothing=random.uniform(-180, 180)
            

            # vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
            #                     .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) 
            #                     for vehicle in self.vehicles]
            vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform) 
                                for vehicle in self.vehicles]
            
            for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
                if not response.error:
                    self.vehicles[i].set_actor(response.actor_id)
                    vehicle_actor = self.vehicles[i].get_actor()
                    #fuck self.trafficmanager.keep_right_rule_percentage(vehicle_actor, 100)
                    #fuck self.trafficmanager.ignore_lights_percentage(vehicle_actor, 100)
                    #fuck self.trafficmanager.ignore_signs_percentage(vehicle_actor, 100)
                    #fuck self.trafficmanager.distance_to_leading_vehicle(vehicle_actor, 5)  # Important for collision avoidance
                    #fuck self.trafficmanager.vehicle_percentage_speed_difference(vehicle_actor, -20)
                    #fuck self.trafficmanager.auto_lane_change(vehicle_actor, True)
                    # If you want the vehicles to avoid each other and the ego vehicle
                    #fuck self.trafficmanager.ignore_vehicles_percentage(vehicle_actor, 0)
                else:
                    print("生成车的地方发生碰撞了:")
                    print(response.error)
                    # 调试：检查清理前后的车辆数量
                    
                    self.world.tick()
                    actors_before = len(self.world.get_actors().filter('vehicle.*'))
                    print(f"清理前世界中的车辆数量: {actors_before}")
                    self.destroy_scene()
                    self.clear_all_vehicles(self.world)
                    self.world.tick()

                    actors_after = len(self.world.get_actors().filter('vehicle.*'))
                    print(f"清理后世界中的车辆数量: {actors_after}")
                    reGenerate=True
                    spawn_points_number= self.count % len(self.spawn_points)
                    ego_location={attr:getattr(self.spawn_points[spawn_points_number].location,attr) for attr in ["x","y","z"]}
                    ego_rotation={attr:getattr(self.spawn_points[spawn_points_number].rotation,attr) for attr in ["yaw","pitch","roll"]}
                    self.yaw =ego_rotation["yaw"]
                    self.count+=1 
                    # ego_location["z"]=0.015
                    self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
                    self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
                    self.ego_vehicle.spawn_actor()
                
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))

        # for vehicle in self.vehicles:
        #     #fuck self.trafficmanager.set_path(vehicle.get_actor(),vehicle.path)

        self.walkers = [Walker(world=self.world,**walker_config) for walker_config in scene_config["walkers"]]
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        # if self.vehicles is not None:
        #     for sensor in self.sensors:
        #         sensor.add_vehicle(self.vehicles[0])
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))
    def generate_custom_scene(self,scene_config):
        
        if scene_config["weather_mode"] == "custom":
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])
        
        self.world.set_weather(self.weather)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])
        ego_bp_name=scene_config["ego_vehicle"]["bp_name"]
        ego_location={attr:getattr(self.spawn_points[self.count].location,attr) for attr in ["x","y","z"]}
        ego_rotation={attr:getattr(self.spawn_points[self.count].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.count+=1 
        ego_location["z"]=0.015

        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        self.ego_vehicle.get_actor().set_autopilot(True, self.trafficmanager.get_port())
        #设计了对自车的驾驶策略
        self.trafficmanager.set_random_device_seed(self.random_seed)
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),0)
        # self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0)
        self.trafficmanager.keep_right_rule_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),5)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)
        traffic = self._get_traffic_config(scene_config)
        if traffic is not None:
            self.vehicles = self._spawn_traffic_vehicles(traffic, ego_location)
            self.walkers = self._spawn_walkers(traffic["walkers"], ego_location=ego_location)
        else:
            self.vehicles = self._spawn_configured_vehicles(scene_config.get("vehicles") or [])
            self.walkers = self._spawn_configured_walkers(scene_config.get("walkers") or [])

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        # if self.vehicles is not None:
        #     for sensor in self.sensors:
        #         sensor.add_vehicle(self.vehicles[0])
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))

    def tick(self):
        self.world.tick()

    def generate_random_scene(self,scene_config):
        if scene_config["weather_mode"] == "custom":
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])
        
        self.world.set_weather(self.weather)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])
        ego_bp_name=scene_config["ego_vehicle"]["bp_name"]
        ego_location={attr:getattr(self.spawn_points[self.count].location,attr) for attr in ["x","y","z"]}
        ego_rotation={attr:getattr(self.spawn_points[self.count].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.count+=1 
        ego_location["z"]=0.015

        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        self.ego_vehicle.get_actor().set_autopilot(True, self.trafficmanager.get_port())
        #设计了对自车的驾驶策略
        self.trafficmanager.set_random_device_seed(self.random_seed)
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),0)
        # self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0)
        self.trafficmanager.keep_right_rule_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),5)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)
        traffic = self._get_traffic_config(scene_config)
        if traffic is not None:
            self.vehicles = self._spawn_traffic_vehicles(traffic, ego_location)
            self.walkers = self._spawn_walkers(traffic["walkers"], ego_location=ego_location)
        else:
            self.vehicles = self._spawn_configured_vehicles(scene_config.get("vehicles") or [])
            self.walkers = self._spawn_configured_walkers(scene_config.get("walkers") or [])

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        # if self.vehicles is not None:
        #     for sensor in self.sensors:
        #         sensor.add_vehicle(self.vehicles[0])
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))     

    def destroy_scene(self):
        # try:
        if self.sensors is not None:
            for sensor in self.sensors:
                if sensor.get_actor() is not None and sensor.get_actor().is_alive:
                    sensor.destroy()
        if self.ego_vehicle is not None and self.ego_vehicle.get_actor() is not None and self.ego_vehicle.get_actor().is_alive:
            self.ego_vehicle.destroy()
        if self.walkers is not None:
            for walker in self.walkers:
                if walker.controller is not None and walker.controller.is_alive:
                    walker.controller.stop()
                if walker.get_actor() is not None and walker.get_actor().is_alive:
                    walker.destroy()
        if self.vehicles is not None:
            for vehicle in self.vehicles:
                if vehicle.get_actor() is not None and vehicle.get_actor().is_alive:
                    vehicle.destroy()
            
        # except Exception as e:
            # print(f"Error in destroy_scene: {e}")
            # exit()


    def destroy_world(self):
        self.trafficmanager.set_synchronous_mode(False)
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None
        self.world.apply_settings(self.original_settings)

    def get_calibrated_sensor(self,sensor):
        sensor_token = generate_token("sensor",sensor.name)
        channel = sensor.name
        if sensor.bp_name == "sensor.camera.rgb" or sensor.bp_name == "sensor.camera.semantic_segmentation":
            intrinsic = get_intrinsic(float(sensor.get_actor().attributes["fov"]),
                            float(sensor.get_actor().attributes["image_size_x"]),
                            float(sensor.get_actor().attributes["image_size_y"])).tolist()
            rotation,translation = get_nuscenes_rt(sensor.transform,"zxy")
        else:
            intrinsic = []
            rotation,translation = get_nuscenes_rt(sensor.transform)
        return sensor_token,channel,translation,rotation,intrinsic
        
    def get_ego_pose(self,sample_data):
        timestamp = transform_timestamp(sample_data[1].timestamp)
        rotation,translation = get_nuscenes_rt(sample_data[0])
        return timestamp,translation,rotation
    def get_vehicle_transform(self,sample_data):
        if len(sample_data) == 3:
            rotation,translation = get_nuscenes_rt(sample_data[2])
            return translation,rotation
        else:
            
            return "",""
    def get_sample_data(self,sample_data):
        height = 0
        width = 0
        if isinstance(sample_data[1],carla.Image):
            height = sample_data[1].height
            width = sample_data[1].width
        return sample_data,height,width

    def get_sample(self):
        return (transform_timestamp(self.world.get_snapshot().timestamp.elapsed_seconds),)
    def get_timestamp(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds
    def get_instance(self,scene_token,instance):
        category_token = generate_token("category",self.category_dict[instance.blueprint.id])
        id = hash((scene_token,instance.get_actor().id))
        return category_token,id

    def get_sample_annotation(self,scene_token,instance):
        instance_token = generate_token("instance",hash((scene_token,instance.get_actor().id)))
        visibility_token = str(self.get_visibility(instance))
        
        attribute_tokens = [generate_token("attribute",attribute) for attribute in self.get_attributes(instance)]
        rotation,translation = get_nuscenes_rt(instance.get_transform())
        # size = [instance.get_size().y,instance.get_size().x,instance.get_size().z]#xyz to whl
        # In client.py get_sample_annotation():
        bbox = instance.get_actor().bounding_box
        # nuScenes size = [width(y), length(x), height(z)] — full dimensions not half
        size = [
            bbox.extent.y * 2,   # width
            bbox.extent.x * 2,   # length  
            bbox.extent.z * 2    # height
        ]
        num_lidar_pts = 0
        num_radar_pts = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                num_lidar_pts += self.get_num_lidar_pts(instance,sensor.get_last_data(),sensor.get_transform())
            elif sensor.bp_name == 'sensor.other.radar':
                num_radar_pts += self.get_num_radar_pts(instance,sensor.get_last_data(),sensor.get_transform())
        return instance_token,visibility_token,attribute_tokens,translation,rotation,size,num_lidar_pts,num_radar_pts

    def get_visibility(self,instance):
        max_visible_point_count = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                ego_position = sensor.get_transform().location
                ego_position.z += self.ego_vehicle.get_size().z*0.5
                instance_position = instance.get_transform().location
                visible_point_count1 = 0
                visible_point_count2 = 0
                for i in range(5):
                    size = instance.get_size()
                    size.z = 0
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count1+=1
                    size.x = -size.x
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count2+=1
                if max(visible_point_count1,visible_point_count2)>max_visible_point_count:
                    max_visible_point_count = max(visible_point_count1,visible_point_count2)
        visibility_dict = {0:0,1:1,2:1,3:2,4:3,5:4}
        return visibility_dict[max_visible_point_count]

    def get_attributes(self,instance):
        return self.attribute_dict[instance.bp_name]

    def get_num_lidar_pts(self,instance,lidar_data,lidar_transform):
        num_lidar_pts = 0
        if lidar_data is not None:
            for data in lidar_data[1]:
                point = lidar_transform.transform(data.point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_lidar_pts+=1
        return num_lidar_pts

    def get_num_radar_pts(self,instance,radar_data,radar_transform):
        num_radar_pts = 0
        if radar_data is not None:
            for data in radar_data[1]:
                point = carla.Location(data.depth*math.cos(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.azimuth)
                        )
                point = radar_transform.transform(point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_radar_pts+=1
        return num_radar_pts

    def get_random_weather(self):
        weather_param = {
            "cloudiness":clamp(random.gauss(0,30)),
            "sun_azimuth_angle":random.random()*360,
            "sun_altitude_angle":random.random()*120-30,
            "precipitation":clamp(random.gauss(0,30)),
            "precipitation_deposits":clamp(random.gauss(0,30)),
            "wind_intensity":random.random()*100,
            "fog_density":clamp(random.gauss(0,30)),
            "fog_distance":random.random()*100,
            "wetness":clamp(random.gauss(0,30)),
            "fog_falloff":random.random()*5,
            "scattering_intensity":max(random.random()*2-1,0),
            "mie_scattering_scale":max(random.random()*2-1,0),
            "rayleigh_scattering_scale":max(random.random()*2-1,0),
            "dust_storm":clamp(random.gauss(0,30))
        }
        return weather_param

    