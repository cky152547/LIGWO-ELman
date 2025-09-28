# -*- coding: utf-8 -*-
import joblib   # 用于保存训练好的模型
import pandas as pd   # 创建数据库，导入数据
import numpy as np   # 用于数据分析，数值运算
import seaborn as sns   # 绘图
from sklearn.preprocessing import StandardScaler   # StandardScaler是sklearn.preprocessing模块中的一个类，用于实现特征缩放，即将数据转换为具有标准差为1的标准正态分布。
from sklearn.model_selection import train_test_split   # train_test_split是sklearn.model_selection模块中定义的一个函数，用于将数据集分割为训练集和测试集。
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix   # roc_auc_score用于得到AUC值。roc_curve用于计算不同阈值下的真正例率（True Positive Rate, TPR）和假正例率（False Positive Rate, FPR），这些值可以用来绘制ROC曲线。classification_report提供了一个文本报告，其中包括了主要的分类指标，如精确度（precision）、召回率（recall）、f1分数（f1-score）和支持度（support）。confusion_matrix用于生成混淆矩阵。
import tensorflow as tf   # tensorflow提供了构建和训练神经网络的工具
from tensorflow.keras.models import Sequential   # Sequential是tensorflow.keras.models模块中的一个类，用于创建一个按顺序堆叠的层的神经网络模型
from tensorflow.keras.layers import Input, SimpleRNN, Dense   # Input是输入层的类，用于定义模型的输入占位符，指定输入数据的形状。SimpleRNN是简单循环神经网络层的类，用于处理序列数据，能够捕捉时间序列中的动态特征。Dense是全连接层的类，用于添加一个全连接的神经网络层。
from tensorflow.keras.optimizers import Adam   # Adam优化器的实现，在神经网络的训练过程中用于调整模型的参数，以最小化损失函数。
import random   # 提供生成随机数的函数。
import math   # 用于数学计算的函数和常量。
import matplotlib   # 绘图
matplotlib.use('Agg') # 或者使用Qt5Agg，当你在没有X服务器的服务器上运行Python脚本时，或者在某些自动化脚本中，你可能不希望图表显示在屏幕上，而是希望直接将图表保存为文件。
import matplotlib.pyplot as plt   # 绘图

# 读取数据
data = pd.read_csv(r'D:\(论文图片)InSAR与改进Elman模型地质灾害危险性评价\CS-GWO-Elman危险性评价图\训练集.csv')   # 读取存储在指定路径的CSV文件。
data = data.dropna()   # 移除含有缺失值（NaN）的行。这通常用于数据清洗，以确保数据分析或机器学习模型训练时使用的数据是完整的。

# 提取特征和标签
x_data1 = data.loc[:, ['as_克里', 'de_克里', '降水', 'SPI', '植被', '距道路', 'NDBI_hebin', 'NDVI_hebin', '土壤', '岩性', '坡向', '坡度', 'dem重采']]
y_data1 = data.loc[:, '类型']   # 使用loc索引器

# 对标签进行独热编码
y_data1 = pd.get_dummies(y_data1).values   # 独热编码是一种将分类数据转换为数值数据的方法，它通过创建新的列来表示每个类别，使得每个实例在类别列上只有一个位置是“热”的（即1），其余位置都是“冷”的（即0）。

# 标准化数据
transfer = StandardScaler()   # 用于将数据标准化，使之具有零均值和单位方差，这通常有助于提高机器学习算法的性能。
x_data1 = transfer.fit_transform(x_data1)   # fit_transform方法首先计算数据的均值和标准差（通过fit步骤），然后对数据进行转换（通过transform步骤），使之符合标准正态分布的特性，即均值为0，标准差为1。处理后的数据重新赋值给x_data1。

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_data1, y_data1, random_state=1, train_size=0.7) # train_test_split 函数首先将数据集随机打乱。70%的数据分配给训练集，剩下的30%分配给测试集。

# 将数据调整为RNN所需的三维格式 (样本数, 时间步长, 特征数)
x_train = np.expand_dims(x_train, axis=1)   # 增加数据的维度。
x_test = np.expand_dims(x_test, axis=1)

# 定义LIGWO算法参数
n_wolves = 10  # 狼群数量在GWO中，狼群通常由α、β、δ三只领头狼组成，它们代表了当前搜索中最好的三个解。但是，这里可能扩展了狼群的概念，使用了更多的“狼”来进行搜索，以便提高算法的多样性和搜索能力。
n_iterations = 20  # 迭代次数定义了算法的迭代次数，即算法将执行20次迭代来寻找最优解。
pa = 0.25  # 被发现的概率在CS中，这个概率通常用来决定一个杜鹃鸟（解）是否被发现。如果一个杜鹃鸟的 fitness 值（适应度）比宿主鸟的差，它可能会被发现并被替换。pa值通常在0到1之间，这里设置为0.25意味着有25%的概率。

# 生成初始狼群
def generate_wolves(n_wolves):
    wolves = []
    for _ in range(n_wolves):
        hidden_neurons = random.randint(10, 100)  # 使用 random.randint 函数生成一个介于 10 到 100 之间的随机整数，代表隐藏层神经元的数量。
        learning_rate = random.uniform(0.0001, 0.01)   # 使用 random.uniform 函数生成一个介于 0.0001 到 0.01 之间的随机浮点数，代表学习率。
        wolves.append([hidden_neurons, learning_rate])   # 将包含 hidden_neurons 和 learning_rate 的列表添加到 wolves 列表中。
    return wolves   # 函数返回包含所有狼参数的列表。

# 计算适应度函数
def fitness(hidden_neurons, learning_rate):   #  hidden_neurons为隐藏层神经元的数量，learning_rate为学习率。
    hidden_neurons = int(hidden_neurons)  # 确保hidden_neurons为整数，因为神经网络的神经元数量必须是整数。
    model = Sequential()   # 构建一个顺序（Sequential）神经网络模型的过程。它定义了模型的结构，包括输入层Input、一个简单的循环神经网络（SimpleRNN）层和一个全连接（Dense）输出层。
    model.add(Input(shape=(1, x_train.shape[2])))  # 使用Input层定义输入形状。shape 参数指定了输入数据的形状，这里 x_train.shape[2] 表示输入数据的第三个维度（在二维数据中，这通常是特征的数量）。1 表示批量大小为1，意味着每个输入样本可以是一个单独的特征向量。
    model.add(SimpleRNN(hidden_neurons, activation='relu', return_sequences=False, stateful=False))   # 这是一个基本的循环神经网络层，用于处理序列数据。hidden_neurons 参数指定了该层中的神经元数量。activation='relu' 指定了使用ReLU作为激活函数。return_sequences=False 表示这个RNN层的输出将不返回序列，只返回最后一个输出。stateful=False 表示这个RNN层的状态不会跨批次传递。
    model.add(Dense(y_train.shape[1], activation='softmax'))   # 添加了一个 Dense 全连接层，用作模型的输出层。y_train.shape[1] 指定了输出层中的神经元数量，这通常与目标类别的数量相对应。activation='softmax' 表示使用softmax激活函数，这在多类别分类问题中常用，可以将输出解释为概率分布。
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['AUC'])   # model.compile()是 Keras 模型的 compile 方法，用于配置模型的训练参数。指定了模型使用的优化器为 Adam，它是一种自适应学习率的优化算法。learning_rate 参数设置了 Adam 优化器的学习率，这个值通常在代码的其他部分定义，并作为参数传递给 Adam 类。指定了模型的损失函数为 categorical_crossentropy，这是一个常用于多类别分类问题的损失函数，它衡量的是模型预测的概率分布与真实标签的概率分布之间的交叉熵。指定了模型训练和评估过程中使用的评估指标，这里使用的是 AUC。
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)   # 用于训练模型。x_train为训练数据。y_train为训练标签。epoch指定了训练迭代的次数，每个 epoch 都会遍历一次完整的训练数据集。这里设置为 10，意味着整个数据集将被用于训练 10 次。batch_size指定了每次迭代中用于训练的样本数量，这里是 32，表示每次训练将从 x_train 和 y_train 中随机选择 32 个样本进行梯度下降计算。verbose控制训练过程中的输出信息。verbose=0 表示在训练过程中不输出任何日志信息，而 verbose=1（默认值）会显示进度条和每个 epoch 结束时的准确率等信息，verbose=2 会提供更详细的逐批次（batch-wise）输出。
    gailv1 = model.predict(x_test)   # 对 x_test 中的每个样本进行预测，并将预测结果作为数组存储在 gailv1 中。预测结果可以用于进一步的评估，例如计算模型的准确率、召回率、F1 分数或其他性能指标。
    auc = roc_auc_score(y_test, gailv1, multi_class='ovr')   # 计算AUC值。y_test为真实的目标值。gailv1为模型预测的结果，可以是概率估计或类别预测。multi_class='ovr'指定了多分类问题中计算 AUC 的方法。这种方法适用于多类别分类问题，其中每个类别都被视为与所有其他类别的组合进行比较。
    return auc

# 莱维飞行
def levy_flight(Lambda):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                      (math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, 2)   # 使用 numpy.random.normal 函数生成两个服从正态分布的随机向量 u 和 v，其中 u 的标准差由 sigma1 确定，v 的标准差由 sigma2 确定。这里的 2 表示生成的向量长度为 2。
    v = np.random.normal(0, sigma2, 2)
    step = u / np.power(np.abs(v), 1 / Lambda)   # 莱维飞行的步长 step 是通过将 u 向量除以 v 向量的绝对值的 Lambda 次幂得到的。这种计算方式使得 step 具有重尾分布特性，即有较小的概率生成较大的步长，这有助于在搜索空间中进行长距离探索。
    return step

# LIGWO算法主循环
def cs_gwo():
    wolves = generate_wolves(n_wolves)   # 调用 generate_wolves 函数生成 n_wolves 个狼的参数，返回一个包含狼的参数（如隐藏层神经元数量和学习率）的列表。
    alpha, beta, delta = sorted(wolves, key=lambda wolf: fitness(*wolf), reverse=True)[:3]   # sorted函数用于对可迭代对象的元素进行排序。key 参数指定了一个函数，用于从列表的每个元素中提取一个用于比较的值。lambda 表达式定义了一个匿名函数，它接受一个参数 wolf（代表列表中的一个元素），然后调用 fitness 函数并传入 wolf（使用 *wolf 对列表进行解包）。reverse=True 参数指定了排序的方式，True 表示降序排序，这样适应度最高的候选解会排在前面。[:3]：这是一个切片操作，用于从排序后的列表中取出前三个元素。这些元素代表当前适应度最高的三个候选解。
    best_fitness = fitness(*alpha)   #  * 表示参数解包（argument unpacking）。如果 alpha 是一个包含多个参数的列表或元组，使用 * 可以将 alpha 中的元素作为独立的参数传递给 fitness 函数。
    history = []

    for iteration in range(n_iterations):
        # 莱维飞行扰动
        step_size = levy_flight(1.5)   # 调用 levy_flight 函数并传入参数 1.5（Lambda 值）。
        for i, wolf in enumerate([alpha, beta, delta]):   # enumerate 函数用于将一个可迭代对象组合为一个索引序列，其中每个元素是一个包含两个元素的元组：第一个是索引（i），第二个是可迭代对象中的元素（wolf）。
            if i == 0:
                wolf = [wolf[0] + step_size[0], wolf[1] + step_size[1]]   # 更新列表 [alpha, beta, delta] 中的第一个元素（即 alpha 狼的位置）。wolf[0] + step_size[0] 表示在第一个维度上，将狼的当前位置加上步长向量的第一个分量。wolf[1] + step_size[1] 表示在第二个维度上，将狼的当前位置加上步长向量的第二个分量。
            else:
                wolf = [wolf[0] + step_size[0] * random.random(), wolf[1] + step_size[1] * random.random()]   # 这里使用 step_size 向量的每个分量乘以一个在 [0, 1) 区间内的随机数（由 random.random() 生成），来更新狼的位置。这种方法引入了随机性，有助于算法在搜索空间中进行探索。
            wolves[i] = [max(10, min(int(wolf[0]), 100)), max(0.0001, min(wolf[1], 0.01))]  #  wolves 中索引为 i 的狼的当前位置。max(10, min(int(wolf[0]), 100))：这部分代码确保第一个位置参数（例如隐藏神经元数量）被限制在 10 到 100 的范围内。首先，使用 int(wolf[0]) 将该参数四舍五入到最近的整数，然后 min(int(wolf[0]), 100) 确保它不会超过 100，接着 max(...) 确保它不会低于 10。max(0.0001, min(wolf[1], 0.01))：这部分代码确保第二个位置参数（例如学习率）被限制在 0.0001 到 0.01 的范围内。min(wolf[1], 0.01) 确保学习率不会超过 0.01，而 max(...) 确保它不低于 0.0001。

        # GWO更新机制
        for i in range(n_wolves):   # 循环遍历狼群中的每只狼，n_wolves 是狼群中狼的数量。
            for j in range(2):   # 循环遍历每只狼的两个决策变量（或称为位置参数）。这里的 2 表示每只狼有两个参数，例如在神经网络中可能是隐藏层神经元数量和学习率。
                A = 2 * random.random() - 1   # A 是一个在 [-1, 1] 区间的随机数，用于调整狼群向领头狼移动的步长。
                C = 2 * random.random()   # C 是一个在 [0, 2] 区间的随机数，用于控制 D_alpha、D_beta 和 D_delta 的比例。
                D_alpha = abs(C * alpha[j] - wolves[i][j])   # D_alpha 表示第 i 只狼在第 j 个参数上与α狼（领头狼）之间的距离。C用于调整距离。alpha[j] 是α狼在第 j 个参数上的值。wolves[i][j] 是第 i 只狼当前在第 j 个参数上的值。
                D_beta = abs(C * beta[j] - wolves[i][j])   # D_beta 表示第 i 只狼在第 j 个参数上与β狼之间的距离。
                D_delta = abs(C * delta[j] - wolves[i][j])   # D_delta 表示第 i 只狼在第 j 个参数上与Δ狼之间的距离。
                X1 = alpha[j] - A * D_alpha   # X1 是灰狼基于 α 狼（领头狼）的新位置。
                X2 = beta[j] - A * D_beta   # X2 是灰狼基于β狼的新位置。
                X3 = delta[j] - A * D_delta   # X3 是灰狼基于Δ狼的新位置。
                wolves[i][j] = (X1 + X2 + X3) / 3   # wolves[i][j] 计算三个调整位置的平均值，作为狼群中第 i 只狼在第 j 个参数上的新位置。
                wolves[i][j] = max(10 if j == 0 else 0.0001, min(int(wolves[i][j]), 100 if j == 0 else 0.01))  # 确保在GWO中更新后的狼群位置 wolves[i][j] 保持在预设的参数有效范围内。当 j == 0 时，确保第一个参数（例如神经元数量）是介于 10 到 100 之间的整数。当 j != 0 时，确保第二个参数（例如学习率）是介于 0.0001 到 0.01 之间的浮点数。

        # 发现概率扰动
        if random.random() < pa:   # random.random() 生成一个 [0, 1) 区间的随机浮点数。
            wolves[random.randint(0, n_wolves-1)] = generate_wolves(1)[0]   # random.randint(0, n_wolves-1)生成一个从 0 到 n_wolves-1 之间的随机整数，用作狼群中某只狼的索引。 generate_wolves 函数预期会返回一个包含新狼位置参数的列表，参数 1 表示生成一只新狼。[0] 从返回的列表中取出第一个元素，即新生成的狼的位置。

        # 重新排序并确定新的alpha、beta和delta
        alpha, beta, delta = sorted(wolves, key=lambda wolf: fitness(*wolf), reverse=True)[:3]   # # sorted函数用于对可迭代对象的元素进行排序。key 参数指定了一个函数，用于从列表的每个元素中提取一个用于比较的值。lambda 表达式定义了一个匿名函数，它接受一个参数 wolf（代表列表中的一个元素），然后调用 fitness 函数并传入 wolf（使用 *wolf 对列表进行解包）。reverse=True 参数指定了排序的方式，True 表示降序排序，这样适应度最高的候选解会排在前面。[:3]：这是一个切片操作，用于从排序后的列表中取出前三个元素。这些元素代表当前适应度最高的三个候选解。
        current_best_fitness = fitness(*alpha)   # 计算 alpha 狼的适应度，即排序后适应度最高的狼。
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
        history.append(best_fitness)
        print(f'Iteration {iteration+1}, Best Fitness: {best_fitness}')

    return alpha, history

# 执行LIGWO算法
best_wolf, history = cs_gwo()
print('Best parameters found:', best_wolf)

# 使用最佳参数训练最终模型
best_hidden_neurons, best_learning_rate = best_wolf   # best_hidden_neurons （最佳隐藏层神经元数量）和 best_learning_rate （最佳学习率）是你创建的变量，用于存储解包后的值。
model = Sequential()   # 创建一个新的顺序模型。
model.add(Input(shape=(1, x_train.shape[2])))  # 使用Input层定义输入形状。shape 参数指定了输入数据的形状，这里 x_train.shape[2] 表示输入数据的第三个维度（在二维数据中，这通常是特征的数量）。1 表示批量大小为1，意味着每个输入样本可以是一个单独的特征向量。
model.add(SimpleRNN(int(best_hidden_neurons), activation='relu', return_sequences=False, stateful=False))  # 这是一个基本的循环神经网络层，用于处理序列数据。hidden_neurons 参数指定了该层中的神经元数量。activation='relu' 指定了使用ReLU作为激活函数。return_sequences=False 表示这个RNN层的输出将不返回序列，只返回最后一个输出。stateful=False 表示这个RNN层的状态不会跨批次传递。
model.add(Dense(y_train.shape[1], activation='softmax'))   # 添加了一个 Dense 全连接层，用作模型的输出层。y_train.shape[1] 指定了输出层中的神经元数量，这通常与目标类别的数量相对应。activation='softmax' 表示使用softmax激活函数，这在多类别分类问题中常用，可以将输出解释为概率分布。
model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='categorical_crossentropy', metrics=['AUC'])   # # model.compile()是 Keras 模型的 compile 方法，用于配置模型的训练参数。指定了模型使用的优化器为 Adam，它是一种自适应学习率的优化算法。learning_rate 参数设置了 Adam 优化器的学习率，这个值通常在代码的其他部分定义，并作为参数传递给 Adam 类。指定了模型的损失函数为 categorical_crossentropy，这是一个常用于多类别分类问题的损失函数，它衡量的是模型预测的概率分布与真实标签的概率分布之间的交叉熵。指定了模型训练和评估过程中使用的评估指标，这里使用的是 AUC。
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)   # 用于训练模型。x_train为训练数据。y_train为训练标签。epoch指定了训练迭代的次数，每个 epoch 都会遍历一次完整的训练数据集。这里设置为 50，意味着整个数据集将被用于训练 50 次。batch_size指定了每次迭代中用于训练的样本数量，这里是 32，表示每次训练将从 x_train 和 y_train 中随机选择 32 个样本进行梯度下降计算。validation_split=0.2指定了从训练数据中划分出来用于验证的比例，这里的 0.2 表示 20% 的训练数据将被用作验证集，用于监控模型在看不见的数据上的性能，防止过拟合。verbose控制训练过程中的输出信息。verbose=0 表示在训练过程中不输出任何日志信息，而 verbose=1（默认值）会显示进度条和每个 epoch 结束时的准确率等信息，verbose=2 会提供更详细的逐批次（batch-wise）输出。

# 预测
gailv1 = model.predict(x_test)
ypredict1 = np.argmax(gailv1, axis=1)
y_test_true = np.argmax(y_test, axis=1)

# 模型评估
print('模型AUC值为', roc_auc_score(y_test, gailv1, multi_class='ovr'))
classreport = classification_report(y_test_true, ypredict1)
print(classreport)

# 保存训练模型
joblib.dump(model, 'cs_gwo_elman_model.pkl')

# 可视化部分
# 1. 绘制 ROC 曲线
fpr = {}
tpr = {}
plt.figure()
for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], gailv1[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (area = %0.2f)' % roc_auc_score(y_test[:, i], gailv1[:, i]))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png') # 保存图片
plt.close() # 关闭图表

# 2. 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test_true, ypredict1)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') # 保存图片
plt.close() # 关闭图表

# 3. 实际值和输出值的散点图
plt.figure()
samples = range(len(y_test_true))  # 样本数量作为横坐标
plt.scatter(samples, y_test_true, alpha=0.3, label='Actual Value')
plt.scatter(samples, np.argmax(gailv1, axis=1), alpha=0.3, label='Predicted Value', color='red')
plt.xlabel('Sample Number')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted Probability')
plt.savefig('actual_vs_predicted.png')  # 保存图片
plt.close()  # 关闭图表

print("所有图表已保存为 PNG 文件。")
