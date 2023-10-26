# 2023LLM_Exam
2023 Kaggle LLM_Exam金牌
# Kaggle LLM Science Exam 金牌方案
LLM在Kaggle的第一场首秀，也是我在Kaggle的首秀，辛苦了两个月经历大起大落，从public rk3 shake到金牌区末尾，来复盘一下整场比赛。
感谢我的四位领导@zhengheng、@sayoulala、@chizhu、@wagege，跟着老牌NLP大师们确实学到了很多，成功两个月从零速通Kaggle Master！

## 1.赛题核心
赛题的任务是给定一系列有关数物化生等科学领域的多项单选题，需要用模型判断出正确选项，而赛题的训练数据只给了200条，且允许使用一切外部数据集，这就代表了整场比赛的核心都围绕着“数据”进行。

得益于大量优质的开源工作，赛题在后半段确定了上分重点，就是借助外挂知识库体系来将问题的“参考资料”送给模型，模型在接收“参考资料”和问题的输入后完成“开卷考试”的选择题判断。

所以赛题的重中之重就是如何将更优质、干净、相关、简短的“参考资料”从茫茫知识库海中召回，送给模型判断。有的队伍选择构建了一个更干净、更细致的优质知识库，而我们的选择是通过多路召回再做精排精筛，来将一个比较原始的知识库筛到几百个单词，这两种方式都能有效上分。用一个比较形象的比喻，我们造了一个高精度大炮来从海里捞针，其他队伍把海缩成了一小碗水来捞针。

用一张图比较详细地演示我们的整体方案。


## 2.知识库构建
赛题背景中举办方说他们是从英文wiki百科中抽取了部分知识点，借助GPT3.5来构建的训练集、测试集，那么原始知识库就可以锁定到英文wiki百科上。

得益于很多优秀的开源工作，有三个优质知识库被开源供所有人使用，三个知识库的数据分布、数据格式都不一致，我们并没有在额外补充知识库（复盘的时候觉得自己很蠢，怎么就想不到在这一点继续动手脚）。

说说其他队伍的知识库过滤，在复盘完后发现还是各不相同的，接下来介绍一种我认为简单、好用的知识库构建方法。

(1)拿到原始知识库，通过句向量模型将其中每篇wiki文章编码成Embeddings，这里比较好用的是Sentence-Transformer模型；

(2)将原始知识库的Embeddings经过聚类模型（如KMeans），将其聚为几类，然后拿到训练集200条问题的原始wiki title，选择出将这200个title尽可能多涵盖的类别；

(3)保留类别，或做二次聚类、迭代筛选，最终确定优质知识库集。

## 3.RAG策略

召回策略应该是本次比赛最为重要的点，我们花了几乎所有精力在构建我们的召回策略上。得益于很多种优秀的文本检索技术，我们基于多种方案一共构建了5路召回策略。

这里以我们做的最为详细的一路作为示例。

(1)借助Sentence-Transformer模型，将海量知识库编码成Embed，然后用Faiss的向量检索技术来快速召回Top1000个文章；

(2)将Top1000个文章用BM25或LGBRanker来做第二步的筛选，过滤到Top30文章；

(3)将Top30文章切割成句子，利用Sentence-Transformer模型来召回Top30个句子。

下面详细讲述每一步用到的技术。

1.首先，我们在检索Top1000文章时用到的Sentence-Transformer模型，是我们利用了Simcse技术进一步训练的，这样能大幅度提升模型的文本相似度表征能力，毕竟经过SFT的模型一定会优于无监督模型。其中Simcse里我们用到了一个叫困难样本对比学习的Trick，这也给我们额外带来了0.005的LB提升。Simcse一共LB上分0.015。

2.Sentence-Transformer模型有着非常快速的双塔体系大规模检索的速度优势，但他的缺点也是非常明显的，受限于Bert的512编码长度，他无法将动则成千上万长度的Wiki文章全部编码用来检索，这就造成了很多的信息浪费。

所以我们提出了第二轮筛选，BM25技术是可以有效编码所有文本的，并且精度非常高，在我们的线下测试中Sentence-Transformer模型Top1000然后用BM25过滤到Top30，命中率是有0.935的，而Sentence-Transformer模型直接召回Top30的命中率却只有0.827。	但BM25技术的缺点也非常明显，他运行的太慢了....故我们构建了一个新的筛选规则，利用做问题和召回文章的特征工程，构建N-Gram重复特征来送给LGB，然后由LGB输出文章排序，这就是LGBRanker。在我们的线下测试中，LGBRanker能拥有和BM25一样的命中精度，但速度却快了50%。

而其他路召回，得益于优秀的开源工作，利用到的数据集被洗的非常干净，我们就只是简单地用Sentence-Transformer和TFIDF来做了段落召回。

值得一提的是，@zhengheng不愧是老牌NLP大师，在他的代码管理下，我们大幅度减少了整个召回流程的内存占用和推理速度，最终内存占用足够我们比其他队伍多召回50%的文本，以及TFIDF的召回速度也比其他队伍快了800%。

最终五路召回都会各自得到几十个句子，这里有2种选择：1.再过一轮精排，将5路召回的句子大洗牌，最有可能的放在最前面；2.直接将5路召回各过一次deberta，然后将输出logits加权融合。在我们的实验下，我们发现第二种方案会有更多的鲁棒性，因为其实正确的参考资料也就那么几句话，且都在5路召回里出现过，deberta在各自接收不同的噪音样本下有着各种极端的logits输出，而这样能让5路召回的文本更关注于其共同点。

## 4.分类器策略

实话说我们在分类器上放的精力是非常少的，主要还是忙不过来了。我们在简单尝试LLAMA 7B后，发现他的LB成绩与deberta v3 large相差无几，而推理速度却慢了好几倍。故我们最终只采用了两个deberta来做分类器。

而且在比赛最后一段时间，不晓得是哪出了问题，分类器在最终适应我们5路召回的方案上效果非常差。。

所以我们采用的两个deberta都是一个月前随便训的两个垃圾模型（Sad，这也导致了我们private榜单的惊天shake。

从赛后复盘来看，是LB的少量数据误导了我们的判断，其实LLM 7B明显在private上比deberta有着更佳的鲁棒性，几乎没有shake，由此来看大模型还是非常有效的。而LLM 70B，这么大的模型甚至不能做到在9h内完整地推理一次测试集，故大部分采用70B的队伍应该只推理了一次，这也导致了没办法做模型融合而有了惊天shake。

## 5.一些杂谈

Kaggle的比赛强度果然名不虚传，竞争太激烈了。总体复盘来看我们由于人力的限制和GPU资源的限制，没有对比赛更多方面进一步深入探索，唯一值得说道的就是我们的“高射炮”召回体系。从这场比赛中我也学到了很多，希望大家一起进步！

```
代码顺序:
1. step1_page.py
2. step3_buildindex.py
3. 生成训练集的 get_bm25.py
4.训练 api.py
```
