def get_zone(x_center, frame_width):

    zone_width = frame_width / 3

    if x_center < zone_width:
        return 1

    elif x_center < zone_width * 2:
        return 2

    else:
        return 3