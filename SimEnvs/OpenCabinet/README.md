# Cap PutStuff2Drawer Data

## 获取数据
    python record_sim_episodes.py --task_name sim_put_apple_to_middle_drawer --onscreen_render
其中 `--task_name` 可换为:
`--task_name sim_put_apple_to_middle_drawer
`
`--task_name sim_put_blue_bottle_to_bottom_drawer
`
`--task_name sim_put_banana_to_bottom_drawer
`
`--task_name sim_put_green_bottle_to_bottom_drawer
`
## 四个物块中心位置变化范围
`def SampleRightBoxCenterPosition():
    x_range = [0.4, 0.45]
    y_range = [-0.3, -0.2]
    z_range = [0.08, 0.08]`

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

`def SapleCabinetPose():
    x_range = [0.78, 0.88]
    y_range = [0.2, 0.25]
    z_range = [0.32, 0.32]`

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])

在一个5*10cm的范围内随机生成柜子。柜子抽屉的开合程度是0.2/0.15，bottom的开合程度是0.2，单位米/m

## 任务成功评价标准
当stuff与抽屉底部接触则成功，反之任务失败。

## 训练视角head的位置
		<camera name="front" pos="0.9 0 0.9" resolution="640 480" fovy="78" mode="fixed" euler="0 0.785 1.57"/>

## 场景视图
![screenshot.png](screenshot.png)

## 注意事项
其中苹果的抓取动作容易使得苹果滚动，所以其抓去成功很看运气，
所以做实验的时候可以不做这个数据。
