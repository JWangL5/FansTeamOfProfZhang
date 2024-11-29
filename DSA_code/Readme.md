```**code_explaination**
1.data_loader.py: Converts the JSON file to CSV, filling missing data with median values.
Output: (1) protein_features_filled.csv (2) protein_sequences.csv
2.sequence_embedding.py: Embeds protein sequences using the lightweight ESM2 model into 1280-dimensional tensors. However, the dimensions were too large, so I applied average pooling to reduce them to 33 dimensions. (This results in 802 proteins with 33 features each.)
Output: protein_embeddings.pt (Data dimensions: [802, 33])
3.combine_feature.py: Normalizes protein features and combines them with protein sequence embeddings.
Output: combined_protein_features.pt (Data dimensions: [802, 39])
4.model_definition.py: Defines the transformer model and the DPT model framework.
5.model_LOOCV.py: Implements Leave-One-Out Cross-Validation (LOOCV), where each sample is used as the test set once while the remaining samples are used for training. This avoids data leakage.
Output: dpt_model_loocv.pth (Model trained in 802 iterations)
model_results.csv (Includes six normalized features, true values, predicted values, errors, and test losses from 802 training iterations)
6.results_visualization.py: Visualizes the output results from model_LOOCV.
Visualizations:
(1) test loss distribution plot (Error Distribution)
(2) error heatmap
(3) six scatter plots of protein features
Output:
(1) protein_feature_scatter_plots.png
(2) test_loss_distribution.png
(3) protein_feature_error_heatmap.png

__中文版__
1.data_loader.py: 将json文件转成csv，中位值填补缺失数据. 
得到数据：(1) protein_features_filled.csv (2) protein_sequences.csv
2.sequence_embedding.py: 蛋白质sequence用esm2轻量模型embedding为1280维度的张量，但是有点太大了，我就平均池化，最后变为33维。（就是得到802个蛋白质33个特征的结果）保存在pt文件
输出结果：protein_embeddings.pt     （数据维度：[802, 33]）
3.combine_feature.py: 蛋白质特征归一化与蛋白质序列的embedding整合
输出结果：combined_protein_features.pt   （数据维度：[802, 39]）
4.model_definition.py: 定义的transformer模型，DPT模型框架
5.model_LOOCV.py: 留一交叉验证（每次都只留一个样本作为测试集，其余的样本用于训练，因此 每个样本都会轮流作为测试集，这就避免了数据泄漏问题）
输出结果：dpt_model_loocv.pth（最后一次802次做训练的模型） model_results.csv（包括六个特征的归一化真实值，预测值，误差；802次训练的test loss）
6.results_visulization.py: 将测试model_LOOCV输出结果可视化。
可视化图：1.test loss分布图（Error Distribution） 2.误差热力图 3. 六张蛋白质特征的散点图
输出结果：(1)protein_feature_scatter_plots.png；(2)test_loss_distribution.png ;(3)protein_feature_error_heatmap.png