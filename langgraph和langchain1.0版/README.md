README
用前必看:
Python版本最好是Python 3.11.9 



使用前请先找到目录里的requestment使用pip安装所需要的依赖

按照阿里云官方文档在环境变量中加入通义千问的key

然后替换tianqi_key这个变量后面的的key



数据库表创建代码如下:(含测试用例)



CREATE TABLE attractions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT '景点名称',
    location VARCHAR(255) COMMENT '地理位置',
    capacity BIGINT COMMENT '可容纳人数',
    established_year YEAR COMMENT '建立年份',
    tag_name VARCHAR(50) NOT NULL COMMENT '标签名称（如：自然景观、历史古迹）', -- 移除UNIQUE约束，避免标签无法复用
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '数据创建时间', -- 补全字段名和默认值
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '数据更新时间' -- 自动更新时间
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='景点基本信息表';

ALTER TABLE attractions 
MODIFY COLUMN established_year INT COMMENT '建立年份（支持公元前，用负数表示，如-221表示公元前221年）';

INSERT INTO attractions (name, location, capacity, established_year, tag_name) 
VALUES
('天安门广场', '北京市', 1000000, NULL, '历史古迹'),
('天坛', '北京市', NULL, 1420, '文化遗产'),
('故宫', '北京市', NULL, 1420, '世界遗产'),
('长城', '中国北方', NULL, -221, '自然与文化双重遗产'),  -- 用-221表示公元前221年
('兵马俑', '西安市', 50000, -210, '历史古迹'),
('西湖', '杭州市', 300000, NULL, '自然景观'),
('布达拉宫', '拉萨市', 8000, 641, '宗教文化'),
('拙政园', '苏州市', 10000, 1509, '园林建筑'),
('桂林山水', '桂林市', 500000, NULL, '自然景观'),
('承德避暑山庄', '承德市（河北省）', 30000, 1703, '世界遗产'),
('莫高窟', '敦煌市', 15000, 366, '文化遗产'),
('丽江古城', '丽江市（云南省）', 250000, NULL, '文化遗产'),
('宏村', '黄山市（安徽省）', 8000, 1131, '古村落');

用户表创建如下
SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for user

-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES ('1', 'xiaoma', '123456');
INSERT INTO `user` VALUES ('2', 'xiaozhou', '123456');
INSERT INTO `user` VALUES ('3', 'xiaoliu', '123456');
INSERT INTO `user` VALUES ('4', 'xiaoshang', '123456');
INSERT INTO `user` VALUES ('6', 'xiaozhang', '654321');
INSERT INTO `user` VALUES ('7', 'xiaowang', '147147');
使用前请在main.py和app.py中修改自己的数据库连接配置
# 数据库配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'travel',
    'charset': 'utf8mb4'
}
目录及文件作用

功能介绍:
-file_path:存放需要切分的文档,目前支持pdf , txt ,doc格式
-live2d:存放前端动态小人的js,css和模型文件
-static:存放前端主要代码
-template:存放登陆页面
-app.py:使用fastapi进行前后端交互
-main.py:主要功能实现
-rag测评.py对rag进行评估

工具示例:
1.向量数据库查询:

2数据库查询工具:
可以查询都有哪些景点


3.天气工具



记忆+问题重写
1.调用两个工具并输出记忆
























2.
根据记忆可以直接问那里天气怎么样,程序问题重写然后交给代理查找工具



来源与相关文本块
文本框下面有来源鼠标悬浮在上面可以看到相关文本块


前端介绍
登录页

主页面

