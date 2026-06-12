from car_wrap import MyCar

class testWrap():
    def __init__(self):
        self.my_car = MyCar()
        self.my_car.task.arm.reset()

    def upDown(self):
        self.my_car.task.arm.switch_side(-1)
        self.my_car.task.arm.set(0.75*self.my_car.task.arm.horiz_mid, 0.08)

if __name__ == '__main__':
    test = testWrap()

    test.upDown()