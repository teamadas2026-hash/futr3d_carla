import carla, random
class Actor:
    def __init__(self,world,bp_name,location,rotation,options=None,attach_to=None):
        self.bp_name = bp_name
        self.world = world
        self.blueprint = world.get_blueprint_library().find(bp_name)
        if self.blueprint.id=='vehicle.audi.etron':
            self.blueprint.set_attribute('color', '255,255,255')
        if options is not None:
            for key in options:
                self.blueprint.set_attribute(key, options[key])
        self.transform = carla.Transform(carla.Location(**location),carla.Rotation(**rotation))
        self.attach_to = attach_to
        self.actor = None


    def set_actor(self,id):
        self.actor = self.world.get_actor(id)

    def spawn_actor(self):
        for attempt in range(10):
            try:
                self.actor = self.world.spawn_actor(self.blueprint, self.transform, self.attach_to)
                break
            except RuntimeError:
                # Use pedestrian nav points for walkers, vehicle spawn points for everything else
                if "walker" in self.bp_name or "pedestrian" in self.bp_name:
                    spawn_points = self.world.get_random_location_from_navigation()
                    self.transform = carla.Transform(spawn_points)
                else:
                    spawn_points = self.world.get_map().get_spawn_points()
                    self.transform = random.choice(spawn_points)
        else:
            raise RuntimeError(f"Could not spawn actor {self.bp_name} after 10 attempts")

    def get_actor(self):
        return self.actor
        

    def destroy(self):
        self.actor.destroy()