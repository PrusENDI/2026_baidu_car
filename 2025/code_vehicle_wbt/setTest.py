from vehicle import ArmBase

my_arm = ArmBase()
my_arm.reset()

while True:
    my_arm.set(my_arm.horiz_mid, 0)
    my_arm.set_hand_angle(130)