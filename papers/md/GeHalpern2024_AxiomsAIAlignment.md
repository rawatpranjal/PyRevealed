## Axioms for AI Alignment from Human Feedback

Luise Ge

Washington University in St. Louis g.luise@wustl.edu

## Ariel D. Procaccia

Harvard University arielpro@g.harvard.edu

## Yevgeniy Vorobeychik

## Daniel Halpern

Harvard University dhalpern@g.harvard.edu

## Evi Micha

Harvard University emicha@seas.harvard.edu

## Itai Shapira

Harvard University itaishapira@g.harvard.edu

Junlin Wu

Washington University in St. Louis yvorobeychik@wustl.edu

Washington University in St. Louis junlin.wu@wustl.edu

## Abstract

In the context of reinforcement learning from human feedback (RLHF), the reward function is generally derived from maximum likelihood estimation of a random utility model based on pairwise comparisons made by humans. The problem of learning a reward function is one of preference aggregation that, we argue, largely falls within the scope of social choice theory. From this perspective, we can evaluate different aggregation methods via established axioms, examining whether these methods meet or fail well-known standards. We demonstrate that both the Bradley-Terry-LuceModel and its broad generalizations fail to meet basic axioms. In response, we develop novel rules for learning reward functions with strong axiomatic guarantees. A key innovation from the standpoint of social choice is that our problem has a linear structure, which greatly restricts the space of feasible rules and leads to a new paradigm that we call linear social choice .

## 1 Introduction

The alignment of AI models with human values is widely recognized as a crucial task. A prominent method for this task, reinforcement learning with human feedback (RLHF), has been used in different applications, such as robotics [4, 17] and recommendations [28, 1]. Recently, RLHF has attracted significant attention as a tool for fine-tuning large language models (LLMs) [22, 32, 26]. A typical implementation of RLHF involves learning a reward model using a pre-trained LLM, which is then utilized to fine-tune an existing LLM. During the learning step, human feedback is provided in the form of ordinal comparisons, and a reward function is learned from these. The most common learning method assumes an underlying random utility model such as the Bradley-Terry-Luce (BTL) model [5, 22, 8] and computes a reward function that corresponds to a maximum likelihood estimator for the observed comparisons.

Is this the 'right' way of aggregating individual preferences towards a socially desirable reward function? To answer this question, we draw on social choice theory , a field that studies collective decision making through a mathematical lens [6]. The maximum likelihood estimation approach is in line with a well-established body of work that assumes that different human participants have preferences stemming from noisy estimation of a common ground truth, and the goal is to learn this ground truth as accurately as possible [29]. But this is not the case when it comes to questions of AI

alignment, where individuals can have legitimate differences of opinion rooted in different values or priorities.

We argue that when preferences are truly heterogeneous, the axiomatic approach -which rose to prominence in social choice with the work of Arrow [2] - may be more suitable. This approach analyzes the desirability of aggregation methods by their satisfaction of certain axioms that capture notions of consensus, fairness, and economic efficiency. Specifically, we are interested in the axiomatic properties of aggregation methods that take ordinal preferences as input and output a reward function. We address the following two research questions: What axioms are satisfied by aggregation methods used by existing RLHF algorithms? And are there alternative aggregation methods that offer stronger axiomatic guarantees?

## 1.1 Our Approach

In social choice theory, axioms are typically defined for rules that map rankings over candidates to a single winner (social choice functions) or a ranking of the candidates (social welfare functions). By contrast, we are interested in rules that assign a reward to each candidate. This gap is easy to bridge, though: we simply consider a ranking of the candidates by in descending reward order.

A much more significant gap is that in classical social choice, all relevant candidates appear in the input preferences, whereas in our setting (where candidates correspond, e.g., to prompts and their responses), we are only given preferences over a relatively small set of candidates identified by their (known) features , and we need to generalize from this information. In practice, this entails using a restricted-commonly, parametric-class of reward models which map candidate features to real-valued rewards, and which we fit to existing data.

Specifically, we assume that a linear reward function defined by a parameter vector determines the reward of each candidate by computing the inner product of the parameter vector and the feature vector of the candidate; these modeling choices are consistent with prior and concurrent work [31, 30, 15] and aim to capture the practice of RLHF. 1 Each human participant (henceforth referred to as a voter ) is associated with a parameter vector, which is unknown to us and is used to specify ordinal preferences over the candidates. Our task is to design linear rank aggregation rules , which aggregate rankings induced by these individual linear functions 2 into a collective ranking that is also induced by a linear function; this is a new paradigm in social choice, for which we coin the term linear social choice .

To evaluate linear rank aggregation rules, we adapt fundamental axioms from social choice theory [6]. The first is Pareto optimality (PO) , which requires that if a candidate a is ranked above candidate b in every input ranking, then the resulting ranking should rank a above b . This is seen as a basic requirement and is satisfied by every standard voting method in the classical setting.

The second axiom is pairwise majority consistency (PMC) : If there exists a reward function that generates a ranking where, for each pair of candidates, a majority of voters agree with the ranking, then the resulting ranking should match that ranking. This axiom is an extension of Condorcet consistency to rankings, and is satisfied by some, but not all, standard voting methods in the classical setting.

## 1.2 Our Results

We start by examining, in Section 3.1, a family of loss-based rules that finds a ranking induced by a parameter vector that optimizes a measure of loss; this measure increases for every disagreement with a voter on a pair of alternatives, where the larger the difference in rewards, the larger the penalty. Crucially, by plugging in binary cross-entropy loss we can recover the BTL model. Our first main result is that whenever the loss function is weakly convex and nondecreasing, or strictly convex conditions satisfied by binary cross-entropy loss, as well as, e.g., exponential and hinge loss - the

1 We can represent the reward model as an embedding layer φ ( x ) which is then aggregated linearly to compute the final reward. If we fix the embedding function φ and treat its output as the feature representation of the outcomes, such as prompt-response pairs, the resulting reward model is linear in φ ( x ) ; see, e.g. [31].

2 In practice, typical RLHF datasets consist of pairwise comparisons, not complete rankings. Assuming rankings as input makes our exposition cleaner, and it is not a fundamental limitation, as we discuss in Section 5.

corresponding rule fails both PMC and PO. This result suggests that the prevailing practice of RLHF is flawed from an axiomatic viewpoint.

In Section 3.2, we take a first step towards addressing this shortcoming. We modify the loss-based formulation to focus on majority preferences rather than individual preferences. This modification defines a family of rules that are PMC, but we show that all of them fail PO by establishing an even stronger impossibility result: In stark contrast to the classical setting, any linear rank aggregation rule that depends only on majority preferences must fail PO.

In order to achieve both PO and PMC, we design (in Section 4) a linear rank aggregation rule that we call Leximax Copeland subject to PO . Not only does it satisfy our two main axioms, it also satisfies two additional ones, majority consistency and winner monotonicity .

To summarize, while widely applied rules fail to meet basic axioms, there are alternative methods that are desirable from this viewpoint. Our approach, therefore, provides a discriminative lens through which to evaluate RLHF methods and AI alignment methods more broadly.

## 1.3 Related Work

During the eight months in which we have actively worked on this project (from September 2023 until May 2024)- and especially in the first few months of 2024 - a slew of independent, concurrent papers seeking to build bridges between social choice and RLHF have become publicly available [10, 12, 19, 30, 23, 7, 15, 27, 25]; this surge of interest points, in our view, to the importance of the agenda.

Three of those papers are position papers that conceptually support our work in that they discuss the possibility of applying an axiomatic approach to RLHF [10, 12, 19], although they do not provide any technical results. By contrast, existing technical papers on RLHF do not take an axiomatic approach. Of the concurrent technical papers, the one that is most closely related to ours is that of Siththaranjan et al. [25]. They show, among other results, that the ranking induced by the reward function that the MLE estimator of the Bradley-Terry-Luce Model returns follows the famous Borda count rule when unrestricted reward functions are allowed. In the classical setting, Borda count has strong axiomatic guarantees, including PO (but not PMC). However, it cannot be realized as a linear rank aggregation rule, and it is arguably impractical for RLHF.

Our work builds on an earlier study by Noothigattu et al. [21], which explores the axiomatic properties of reward functions defined as MLE estimators of underlying random utility models. The key difference is that their approach allows for general reward functions, not just linear ones, and they do not consider features at all. Unlike our findings, they show that the BTL model satisfies Pareto Optimality under these conditions. Additionally, they find that pairwise majority consistency is violated even without assuming linearity. However, their results strongly depend on varying the number of comparisons across different pairs of candidates. By contrast, our findings demonstrate that pairwise majority consistency is violated even when the number of comparisons is equal across all pairs of candidates.

## 2 The Linear Social Choice Model

Let C be a set of m distinct prompt/responses, referred to as candidates , and let V = { 1 , . . . , n } be a set of n human participants, known as voters . We denote by R d the d -dimensional real space in which both candidate feature vectors and chosen parameter vectors lie.

Each candidate c ∈ C is associated with a distinct feature vector x c ∈ R d . A parameter vector θ ∈ R d induces a linear reward function r θ : C ↦→ R defined by taking the dot product with feature vectors r θ ( c ) = 〈 θ, x c 〉 . We will primarily be interested in how these parameterized functions rank the candidates by reward. Let R a /follows b = { θ | r θ ( a ) ≥ r θ ( b ) } be the region where the reward of a is at least as large as that of b . Note that R a /follows b and R b /follows a split R d into two half spaces, separated by the hyperplane orthogonal to x a -x b . Parameter vectors θ on the hyperplane have r θ ( a ) = r θ ( b ) , while rankings in the interior of either half-space strictly rank one over the other.

For a ranking σ over the candidates, we say that θ induces σ , denoted θ /triangleright σ , if a /follows σ b implies r θ ( a ) ≥ r θ ( b ) . Let R σ = { θ | θ /triangleright σ } be the set of vectors θ that induce it. Note that this can be written as the intersection of corresponding half spaces R σ = ⋂ a,b : a /follows σ b R a /follows b . Further, the

collection of { R σ } essentially form a partition of R d , covering the space and intersecting only at their boundaries.

/negationslash

We call a θ non-degenerate if it is fully on one side of each of the separating hyperplanes, i.e., r θ ( a ) = r θ ( b ) for all a, b ∈ C . Non-degenerate parameter vectors lie in the interior of some R σ , and thus induce exactly one ranking. We call σ feasible if R σ has a nonempty interior, i.e., is induced by some nondegenerate θ . 3

Each voter i ∈ V submits a ranking over the candidate σ i . We assume that the feature space is rich enough that voter preferences can be captured via non-degenerate parameter vectors. In other words, we assume that each σ i is feasible. We refer to the vector of voter rankings π = ( σ i ) i ∈ V as a profile . Further, for two candidates a, b , we write n a /follows b ( π ) := |{ i ∈ V | a /follows σ i b }| for the number of voters that prefer a to b , and w a /follows b ( π ) = n a /follows b ( π ) /n for the proportion of such voters. When the profile π is clear from context, we may shorten these to n a /follows b and w a /follows b , respectively.

We define a parameter aggregation rule as a function that takes as input a profile π and outputs a parameter vector θ ∗ . Our goal is to design parameter aggregation rules such that r θ ∗ satisfies desirable properties with respect to the voter preferences. However, as the properties we care about will only be with respect to how r θ ∗ ranks the candidates, it will be more convenient to work with what we call linear rank aggregation rules that take as input a profile π and output a feasible ranking σ . There is a natural way to interpret a parameter aggregation rule as a linear rank aggregation rule, namely, output any feasible ranking induced by θ ∗ . The exact properties of the parameter aggregation rule could in principle be sensitive to the tie-breaking of non-degenerate outputs, however, all of our results will be robust to such tie-breaking. 4

We pay special attention to a prominent family of rules from social choice theory referred to as C 1 rules [14], whose outputs depend only on majority relationships, i.e., they only need to know for each pair of candidates ( a, b ) whether the majority prefers a or b .

In our study, we examine several axioms borrowed from social choice theory to evaluate the reasonableness (fairness) of our aggregation mechanisms. These axioms include:

Definition 2.1 (Pareto Optimality) . A linear rank aggregation rule f satisfies Pareto optimality if, whenever every voter prefers candidate a over candidate b on π , i.e., w a /follows b ( π ) = 1 , then candidate a is ranked higher than candidate b in the output ranking, i.e., a /follows f ( π ) b .

Definition 2.2 (Pairwise Majority Consistency (PMC)) . A ranking σ is called a PMC ranking for profile π if for all a, b ∈ C , a /follows σ b if and only if a majority of voters rank a /follows σ i b , i.e., w a /follows b &gt; 1 / 2 . A linear rank aggregation rule satisfies PMC if, when a PMC ranking σ exists for the input profile π and σ is feasible, then f ( π ) = σ .

Note that a PMC ranking for each π need not exist, but when one does, it is unique. The words ' σ is feasible' allude to the possibility that no non-degenerate parameter vector θ induces the unique PMC ranking. Indeed, we have such an example; see Appendix B for details.

Our research question, then, is whether these axioms can be simultaneously satisfied by linear rank aggregation rules. Our approach seeks to provide a concrete illustration of how theoretical insights from social choice can inform practical algorithm design in RLHF.

3 The number of feasible rankings is in general upper bounded by m O ( d ) due to how many regions ( m 2 ) hyperplanes can partition the space into [13]. Further, under mild conditions on the feature vector locations, the exact number of feasible rankings is known [11, 16].

4 One may wonder if there are any computational barriers to converting between parameter and linear rank aggregation rules. However, this is not the case. In particular, for every set of pairwise comparisons R = { a 1 /follows b 1 , a 2 /follows b 2 , a 3 /follows a 3 , . . . } , we can efficiently (i) check if there is a feasible ranking σ consistent with R , and (ii) if such a σ exists, find a nondegenerate θ inducing such a σ . This can be done by finding a θ satisfying the following system of linear inequalities, or determining that no such θ can satisfy them (which can be done using a linear program): r θ ( a ) ≥ r θ ( b )+1 , ∀ a /follows b ∈ R. Any θ satisfying the system would be nondegenerate and induce a σ consistent with R . Furthermore, if a feasible ranking σ is consistent with R , then taking any nondegenerate θ inducing σ and scaling up its values would satisfy all inequalities. This means that given a ranking σ = c 1 /follows c 2 /follows · · · /follows c m , we can check whether or not it is feasible, and so, find a θ inducing it by running this with R = { c 1 /follows c 2 , . . . , c m -1 /follows c m } . Additionally, given a possibly degenerate θ , we can find a feasible ranking σ which θ induces by running this with R = { a /follows b | r θ ( a ) &gt; r θ ( b ) } .

## 3 Loss-Based Rules

## 3.1 Standard Loss Formulation

We begin our study of linear social choice by considering a quite broad yet natural class of rules that capture how RLHF is currently being done. Their core idea is the following: when considering parameter vector θ , for each voter i that ranks a pair of candidates a /follows i b , we should incur some loss for giving b a higher reward than a . To formalize this, let /lscript : R → R be a loss function , which we assume is nonnegative. We can then choose a parameter vector minimizing

/negationslash

<!-- formula-not-decoded -->

Note that the BTL model fits within this framework using /lscript ( x ) = ln(1 + e x ) , i.e., binary crossentropy loss . 5 One caveat to this approach, however, is that an optimal θ need not be well-defined: it is possible that no minimum is attained. Fortunately, since we only care about rankings induced by optimal parameter vectors, we can conveniently remedy this by saying the output is any ranking that is induced by parameter vectors that are arbitrarily close to optimal. More formally, we say that a linear rank aggregation rule f minimizes /lscript if for all σ = f ( π ) ,

<!-- formula-not-decoded -->

Even if no minimum is attained, there is always a choice of feasible ranking σ such that Equation (1) is satisfied.

With this definition in hand, we proceed to our first main result, which spells rather bad news for this class of rules: Any loss-based aggregation rule using a nondecreasing and convex loss function (of which BTL is one, and hinge loss is another) will fail our two core axioms, PMC and PO. This paints a negative picture for current RLHF methods with respect to their social choice guarantees. Note that we will exclude the discussion of loss functions with a global minimum at zero, like ReLU, because the loss minimizer will be zero, making all rankings vacuous consequently. And we have focused on convex loss functions due to their practical optimization ease.

Theorem 3.1. If a linear rank aggregation rule f optimizes a loss function /lscript that satisfies inf x /lscript ( x ) &lt; /lscript (0) and is either nondecreasing and weakly convex, or strictly convex (and possibly nonmonotone), then f fails PMC and PO.

/negationslash

Proof. Fix a loss function /lscript satisfying the theorem conditions. Note that since /lscript is convex, we may also assume it is continuous [24, Corollary 10.1.1]. Furthermore, since inf x /lscript ( x ) &lt; /lscript (0) , we know that there exists x = 0 such that /lscript ( x ) &lt; /lscript (0) . The case where x &gt; 0 is relatively simple (as such loss functions lead to unnatural behavior), and we handle it at the end of the proof. For now, we assume that there exists x &lt; 0 such that /lscript ( x ) &lt; /lscript (0) . Note that this also implies that for all y ≥ 0 , /lscript is lower bounded by the affine linear function connecting ( x, /lscript ( x )) and (0 , /lscript (0)) , and thus, lim x →∞ /lscript ( x ) = ∞ .

We begin with a small instance of just three candidates C core = { a, b, c } to gain some traction on how /lscript behaves. We will later extend this instance with additional candidates to demonstrate a profile where PO and PMC fail. The candidates will have feature vectors x a := (2 , 1) , x b := (1 , 1) , and x c := (0 , 0) , respectively. Furthermore, a p -fraction of voters (for p to be chosen later) will rank a /follows b /follows c , while the remaining (1 -p ) -fraction will have inverted preferences, ranking c /follows b /follows a . 6

Let

<!-- formula-not-decoded -->

/negationslash

Typically, BTL is presented as choosing θ maximizing the likelihood of seeing the pairwise comparisons we observed, assuming that Pr[ a /follows b ] = e r θ ( a ) a b . That is, we choose θ maximizing

/negationslash

∏ a = b ( e r θ ( a ) e r θ ( a ) + e r θ ( b ) ) a /follows b . By taking the log and swapping the sign, we see that this is equivalent to minimizing a b log 1 + e r θ ( b ) -r θ ( a ) .

/negationslash

6 It so happens that these rankings are feasible, but for now we will not worry about this as loss functionbased rules still make sense regardless of whether the inputs are feasible. For the final example, we will ensure that the rankings are feasible.

5 e r θ ( ) + e r θ ( ) n ∑ = ( )

with w x /follows y ∈ { 1 -p, p } be the loss function on this instance (scaling n x /follows y down to w x /follows y leads to an equivalent formulation). Let g ( x ) = p · /lscript ( -x ) + (1 -p ) /lscript ( x ) . Note that we can rewrite L com as

<!-- formula-not-decoded -->

Note that r θ ( c ) = 0 for all θ , so we can simplify this to

<!-- formula-not-decoded -->

We will consider an unconstrained version of this problem where we are free to choose rewards r a , r b ∈ R arbitrarily, and later show by which vectors θ these optimal values can be induced. That is, we will first find r a , r b ∈ R minimizing

<!-- formula-not-decoded -->

Let OPT core = { θ | L core ( θ ) = inf θ ′ L core ( θ ′ ) } and OPT unconstr { ( r a , r b ) | L unconstr ( r a , r b ) = inf r ′ a ,r ′ b L unconstr ( r ′ a , r ′ b ) } be the set of minimizers for these two loss functions. In Appendix A, we establish the following results about these optimal sets.

Lemma 3.2. There exists a rational p ∈ (1 / 2 , 1] and values A 1 &lt; A 2 with A 2 &gt; 0 such that OPT unconstr is nonempty and for all ( r a , r b ) ∈ OPT unconstr , r a &gt; A 2 and r b ≤ A 1 .

Lemma 3.3. Suppose Lemma 3.2 holds for values p, A 1 and A 2 , then, for this same choice of p , OPT core is nonempty and there exist A 3 and A 4 with A 3 &gt; 0 such that for all ( θ 1 , θ 2 ) ∈ OPT core , θ 1 &gt; A 3 and θ 2 &lt; A 4 .

We will now explicitly construct a family of instances with candidate feature vectors parameterized by a value ε ∈ R such that for sufficiently small ε &gt; 0 , the output of f fails the two axioms. Fix p, A 3 and A 4 from Lemma 3.3, and choose δ with 0 &lt; δ &lt; 1 such that δA 4 -A 3 &lt; 0 ( δ &lt; A 3 /A 4 works if A 4 &gt; 0 , and otherwise, any 0 &lt; δ &lt; 1 will do).

Each instance will have six candidates, which we will think of as two groups of three, C = C core ∪ C copies . The first group C core = { a, b, c } will be the same as the three-candidate instance from above, while the second group C copies = { a ′ , b ′ , c ′ } will be new. The candidates a, b, c will still be located at x a := (2 , 1) , x b := (1 , 1) , and x c := (0 , 0) , respectively. The candidates a ′ , b ′ , c ′ will be located near their undecorated counterparts at x a ′ := x a + ( -ε, 0) , x b ′ := x b + ( -ε, 0) and x c ′ := x c +( -ε, δ · ε ) .

Next, we describe the voter preferences. A p -fraction of voters will have the ranking a /follows a ′ /follows b /follows b ′ /follows c ′ /follows c , and the remaining (1 -p ) -fraction of voters will have ranking c ′ /follows c /follows b ′ /follows b /follows a ′ /follows a . As long as 0 &lt; ε &lt; 1 (which will be the case for our final chosen ε ), these are both feasible rankings. The former is induced by the nondegenerate feature vector (1 , 1) 7 and the latter by ( -1 , 0) . 8

For each ε ∈ R , let

/negationslash

<!-- formula-not-decoded -->

with w x /follows y ∈ { 0 , 1 -p, p, 1 } be the loss function we are optimizing using candidate locations parameterized by ε .

We will show that for sufficiently small ε &gt; 0 , inf θ ∈ R c ′ /follows c L ε ( θ ) &gt; inf θ L ε ( θ ) . This means that f must output a ranking with c /follows c ′ . Observe that this is a PO violation because all voters agree that c ′ /follows c . Furthermore, this is a PMC violation because a majority of voters have the ranking a /follows a ′ /follows b /follows b ′ /follows c ′ /follows c , yet this is not the output.

Let OPT ( ε ) be the set of vectors optimizing L ε . The rest of the proof will follow from the following two lemmas, whose proofs are in Appendix A.

Lemma 3.4. OPT (0) ⊆ R c ′ /follows c .

Lemma 3.5. Suppose OPT (0) ⊆ R c ′ /follows c , then, for sufficiently small ε &gt; 0 , inf θ : θ ∈ R c ′ /follows c L ε ( θ ) &gt; inf θ L ε ( θ ) .

7 This induces rewards 3 , 3 -ε, 2 , 2 -ε, (1 -δ ) · ε and 0 for a, a ′ , b, b ′ , c ′ , and c , respectively.

8 This induces rewards ε, 0 , -1 + ε, -1 , -2 + ε and -2 . for c ′ , c, b ′ , b, a ′ , and a , respectively.

Finally, we handle the case that exists x &gt; 0 such that /lscript ( x ) &lt; /lscript (0) . Note that by convexity, this implies that for all y &lt; 0 , /lscript ( y ) &gt; /lscript (0) &gt; /lscript ( x ) , so inf y ≤ 0 /lscript ( y ) &lt; inf y /lscript ( y ) . Now, consider an instance with two candidates { a, b } located at x a = (1 , 0) and x b = (0 , 1) , and a single voter ranking a /follows b (feasible via the parameter vector (1 , 0) ). It is possible to achieve a loss of /lscript ( x ) , e.g., by outputting the parameter vector (0 , x ) . On the other hand, any θ inducing the ranking a /follows b will be lower bounded by /lscript (0) &gt; /lscript ( x ) from above. Hence, f must output b /follows a , which is both a PO and PMC violation.

## 3.2 Majority-Based Loss Formulation

Despite the negative results for loss-function-based rules, we may hope for a remedy using slightly different information. Specifically, we consider a similar loss-based function that rather than getting penalized for disagreeing with each voter only gets penalized if it disagrees with a majority of voters. That is, we choose θ minimizing

/negationslash

<!-- formula-not-decoded -->

Defining a parameter aggregation function based on this loss suffers from the same caveat as before, that in some cases no optimal θ exists. Nevertheless, we can apply an analogous fix for a ranking variant. We say that a linear rank aggregation rule f minimizes /lscript in the majority formulation if for all σ = f ( π ) ,

<!-- formula-not-decoded -->

We first show (in Appendix A.5) that this does indeed help achieve PMC with essentially all loss functions.

Theorem 3.6. Fix a nondecreasing loss function /lscript with /lscript (0) &gt; inf x /lscript ( x ) . If a linear rank aggregation rule f minimizes /lscript in the majority formulation, then f satisfies PMC.

Note that if the /lscript (0) &gt; inf x /lscript ( x ) condition is not satisfied, i.e., /lscript (0) = inf x /lscript ( x ) , then all linear rank aggregation rules f minimize /lscript in the majority formulation, so satisfying this is a vacuous condition. Indeed, the parameter vector 0 of all 0 s achieves optimal loss of /lscript (0) for each pair and is consistent with every ranking σ . Therefore, the condition /lscript (0) &gt; inf x /lscript ( x ) is as innocuous as possible to rule out these edge cases.

However, despite this good news for PMC, we show that this does not help in achieving PO. In fact, our negative result extends to every C 1 linear rank aggregation rule. Note that if f minimizing /lscript in the majority formulation breaks ties consistently (i.e., if multiple feasible rankings are optimal, then it consistently chooses the same one), then it is C1. We then have the following result, whose proof is relegated to Appendix A.6.

Theorem 3.7. All C 1 linear rank aggregation rules fail PO.

This result is quite unfortunate, because if there were a rule that is both C 1 and PO, we would automatically achieve PMC: Whenever there is a feasible PMC ranking, a C 1 rule cannot distinguish between this profile and a profile where all voters submit this ranking, hence, under the PO criterion, it must output it. Furthermore, whenever there is a PMC ranking, outputting it is necessarily PO, as for every pair, a majority of voters agree with the PMC ranking. Interestingly, in the proof, we construct a profile which has a PMC ranking, yet it is not feasible, and no matter how a C 1 linear rank aggregation rule breaks ties, there is an underlying profile in which this output violates PO.

## 4 Social Choice-Based Rule

In light of the above negative results, in this section, we ask whether there are linear rank aggregation rules that concurrently satisfy our two core axioms, PO and PMC. We answer this question affirmatively by presenting a new method based on a prominent rule from voting theory.

The Copeland rule assigns a Copeland score to each alternative equal to the number of other alternatives it beats in a pairwise competition, i.e., the score for a is |{ b | w a /follows b &gt; 1 / 2 }| . It then ranks the candidates in descending order according to their Copeland scores (breaking ties arbitrarily). It

is known that Copeland satisfies PO, PMC, and additional axiomatic properties. However, in linear social choice, since not every ranking is feasible, we cannot always output the Copeland ranking.

We, therefore, define a new linear rank aggregation rule, which we call leximax Copeland . This rule chooses a feasible ranking as follows. It ranks first the candidate with the highest Copeland score that can be feasibly ranked first under some parameter vector θ . Subject to this first position, it ranks second the candidate with the highest Copeland score which can be feasibly ranked second, and continues this process for subsequent positions.

Copeland's rule is a C 1 rule because it only requires the majority relationships between the candidates. Analogously, leximax Copeland is also a C 1 linear rank aggregation rule. Therefore, by Theorem 3.7, it does not satisfy the PO criterion. To address this issue, we define a variant called leximax Copeland subject to PO (LCPO) , which incorporates the PO criterion. Under LCPO, for every pair of alternatives where one dominates the other, the rule restricts rankings to place the dominating alternative above the dominated one.

The rule remains well-defined since the set of feasible rankings when enforcing the PO criterion is non-empty, as whenever a dominates b , all the rankings in the input profile rank a above b . Note that if the Copeland ranking is feasible, then this rule outputs that ranking, since unrestricted Copeland satisfies PO.

In addition to PO and PMC, we wish to show that LCPO satisfies two additional properties, which we define presently.

Definition 4.1 (majority consistency) . A linear rank aggregation rule satisfies majority consistency if when a candidate a is ranked first by a majority of voters in the input profile, a is ranked first in the output ranking.

Majority consistency ensures that the collective decision reflects the preference of the majority when there is a clear favorite. This principle aligns with PMC, but specifically focuses on the majority's favorite alternative. However, as we discussed above, a PMC ranking does not necessarily exist, and even when it exists, it is not necessarily feasible. By contrast, when a majority winner exists, this candidate is necessarily ranked first by a majority of voters in the input profile, who themselves (by assumption) submit feasible rankings. Therefore, we need not handle the case where it is impossible to rank the majority winner first.

Definition 4.2 (winner monotonicity) . A linear rank aggregation rule satisfies winner monotonicity if, when a candidate a is ranked first in the output ranking, elevating a in any voter's preference does not cause a to lose their top position in the updated aggregate ranking.

Winner monotonicity ensures that improving a leading candidate's position among individual voters will not result in that candidate's demotion.

We now state and prove the main result of this section.

## Theorem 4.3. LCPO satisfies PO, PMC, majority consistency and winner monotonicity.

Proof. LCPO trivially satisfies PO since it always outputs a ranking that respects the PO criterion. Moreover, since Copeland satisfies PMC, and whenever Copeland's ranking is in the domain, leximax Copeland subject to PO returns this ranking, it clearly satisfies PMC.

Note that if an alternative a is ranked first by at least half of the voters, then a has the highest Copeland score, meaning that leximax Copeland subject to PO will rank this candidate first if this is possible. We see that this is indeed possible, by noticing that there is at least one feasible ranking in the input profile where a is ranked first, and any such input ranking is feasible (by assumption) and satisfies the PO requirement. Therefore, majority consistency is satisfied.

It remains to show that LCPO satisfies winner monotonicity. Suppose that on input profile π , the rule outputs a ranking σ where candidate a is ranked first. Now, consider a profile π ′ which is similar to π with the only exception being a ranking in which a is placed in a higher position. Let S be the set of agents that ranked above a in Copeland's ranking under π and let S ′ be the set of agents that ranked above a in Copeland's ranking under π ′ . Note that S ′ ⊆ S , since when moving from π to π ′ , only the Copeland score of a can increase, and therefore it is not possible for a candidate b to beat a under π ′ but not under π .

Now, suppose that R and R ′ are the set of rankings that satisfy the PO criterion with respect to π and π ′ , respectively. We show that R ′ ⊆ R . First, note that since a is ranked first under π , no alternative dominates a in π , as otherwise the PO criterion would be violated. Therefore, we get that no other alternative dominates a in π ′ as well. Moreover, note that if b dominates c in π , then this remains true in π ′ as well. On the other hand, it is possible that a dominates an alternative b in π ′ but not in π . From all of the above, we conclude that R ′ ⊆ R .

Since a is ranked first in σ , we get that for every candidate b ∈ S , there is no ranking in R in which b is ranked first, since otherwise, LCPO would output such a ranking. This also means that for every candidate b in S ′ , there is no ranking in R ′ in which b is ranked first, since R ′ ⊆ R and S ′ ⊆ S . Moreover, note that every ranking in R in which a is ranked first is also in R ′ since it satisfies all the PO restrictions of π ′ . Therefore, under π ′ , LCPO outputs a ranking in which a is ranked first.

Leximax Copeland subject to PO can be implemented in polynomial time by solving O ( | C | 2 ) relatively small linear programs. Specifically, given an input profile, we sequentially choose the candidate that is ranked in position r +1 as follows. We denote by σ r the partial ranking, where the first r positions have been fixed. For each candidate c that has not been ranked yet, we want to check if there is a parameter vector that adheres to the partial ranking σ r , respects the Pareto optimality criterion and ranks c at position r + 1 . Since all these constraints can be expressed as pairwise comparisons, we can use a linear program such as the one described in Footnote 4 to check if such a feasible ranking exists. Among the candidates meeting this criterion, we select the one with the highest Copeland score for position r .

## 5 Discussion

We conclude with a discussion of several extensions and limitations of our approach and results.

First of all, we wish to emphasize that our results are theoretical. While they highlight some shortcomings of the current practice of RLHF, our goal was not to 'outperform' existing RLHF methods. Rather, we see our model as giving a framework for understanding and comparing rules and methods-it is a (useful, we believe) lens through which researchers and engineers can examine their AI alignment methods.

Second, as written, our model has voters give their complete rankings, while in practice, this would be infeasible. In the real world, we are likely to elicit only relatively few pairwise comparisons per person. For our negative results, this assumption only makes them stronger: the BTL model fails both PO and PMC even with access to complete voter rankings. By contrast, for the positive results, specifically implementing leximax Copeland subject to PO, this ostensibly seems like a serious limitation. However, the complete rankings are not necessary for computing this rule, rather, all we need to know are PO dominance relationships and majority directions. We can therefore apply the rule whenever we can approximate this information, for example, through sampling. An alternative approach is to infer a complete ranking of each voter by fitting a parameter vector based on their pairwise responses; this process of learning a complete ranking and then running voting rules has been used before in a variety of settings [20, 18].

Third, our work initiates the study of the axiomatic method in our linear social choice model. However, we leave open many questions about which axioms are compatible and finding rules that achieve them. It should be clear by now that the primary challenge in linear social choice is that not every ranking over the candidates can be output. This means that essentially all known aggregation rules cannot directly be used without at least some modification. A natural direction to tackle is to try to find methods of converting known voting rules into linear aggregation ones while maintaining some of their axiomatic properties. To this end, we conclude with some preliminary results, and somewhat surprising findings within this space.

Some rules which optimize over rankings can be naturally transformed. For example, consider the Kemeny rule, which returns the ranking with the smallest pairwise disagreement over all votes. This can easily be transformed to the linear setting by simply outputting the optimal feasible ranking. In fact, in Appendix C.2, we show that this rule carries over the property of separability , 9 a social choice axiom that is violated by Copeland (in the classical setting) and leximax Copeland subject to

9 Formally, ranking separability to distinguish it from the single-winner version.

PO (in our setting). We show this in Appendix C.1. However, quite strikingly, although separability remains, this transformation makes Kemeny no longer PO (Appendix C.2).

Finally, note that the 'leximax' portion of leximax Copeland can be seen as a general purpose tool for mapping traditional rules to linear aggregation rules. In Appendix C.3, we explore leximax plurality (run leximax on the ranking of candidates by plurality scores), and show that it satisfies majority consistency, winner monotonicity, and separability. Additionally, the 'subject to PO' can be seen as another 'tool' for enforcing the Pareto optimality criterion when a rule does not independently satisfy it. However, enforcing PO can again cause somewhat surprising results. For example, in Appendix C.2, we show that linear Kemeny subject to PO, while now trivially satisfying PO, again violates separability. These observations indicate the challenges inherent in linear social choice, and we hope these open questions inspire fruitful follow-up research.

## Acknowledgments

This research was partially supported by the National Science Foundation under grants IIS-2147187, IIS-2229881, CCF-2007080, IIS-1905558, and IIS-2214141; and by the Office of Naval Research under grants N00014-20-1-2488 and N00014-24-1-2704.

## References

- [1] Nir Ailon and Mehryar Mohri. Preference-based learning to rank. Machine Learning , 80: 189-211, 2010.
- [2] Kenneth Arrow. Social Choice and Individual Values . Wiley, 1951.
- [3] Claude Berge. Topological Spaces . Oliver and Boyd, 1963.
- [4] Erdem Bıyık, Nicolas Huynh, Mykel J Kochenderfer, and Dorsa Sadigh. Active preferencebased Gaussian process regression for reward learning. arXiv preprint arXiv:2005.02575 , 2020.
- [5] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika , 39(3/4):324-345, 1952.
- [6] Felix Brandt, Vincent Conitzer, Ulle Endriss, Jerˆ ome Lang, and Ariel D. Procaccia, editors. Handbook of Computational Social Choice . Cambridge University Press, 2016.
- [7] Souradip Chakraborty, Jiahao Qiu, Hui Yuan, Alec Koppel, Furong Huang, Dinesh Manocha, Amrit Singh Bedi, and Mengdi Wang. MaxMin-RLHF: Towards equitable alignment of large language models with diverse human preferences. arXiv preprint arXiv:2402.08925 , 2024.
- [8] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems , 30, 2017.
- [9] F.H. Clarke. Optimization and Nonsmooth Analysis . Wiley New York, 1983.
- [10] Vincent Conitzer, Rachel Freedman, Jobst Heitzig, Wesley H Holliday, Bob M Jacobs, Nathan Lambert, Milan Moss´ e, Eric Pacuit, Stuart Russell, Hailey Schoelkopf, Emanuel Tewolde, and William S. Zwicker. Social choice for AI alignment: Dealing with diverse human feedback. arXiv preprint arXiv:2404.10271 , 2024.
- [11] Thomas M Cover. The number of linearly inducible orderings of points in d -space. SIAM Journal on Applied Mathematics , 15(2):434-439, 1967.
- [12] Jessica Dai and Eve Fleisig. Mapping social choice theory to RLHF. arXiv preprint arXiv:2404.13038 , 2024.
- [13] H. Edelsbrunner. Algorithms in Combinatorial Geometry , volume 10 of EATCS Monographs on Theoretical Computer Science . Springer, 1987.
- [14] Peter C Fishburn. Condorcet social choice functions. SIAM Journal on applied Mathematics , 33(3):469-489, 1977.
- [15] Luise Ge, Brendan Juba, and Yevgeniy Vorobeychik. Learning Linear Utility Functions From Pairwise Comparison Queries. arXiv preprint arXiv:2405.02612 , 2024.
- [16] Henry W Gould. A note on the number of linearly inducible orderings of points in d -space. SIAM Journal on Applied Mathematics , 26(3):528-530, 1974.
- [17] Andras Kupcsik, David Hsu, and Wee Sun Lee. Learning dynamic robot-to-human object handover from human feedback. Robotics Research: Volume 1 , pages 161-176, 2018.
- [18] Min Kyung Lee, Daniel Kusbit, Anson Kahng, Ji Tae Kim, Xinran Yuan, Allissa Chan, Daniel See, Ritesh Noothigattu, Siheon Lee, Alexandros Psomas, et al. Webuildai: Participatory framework for algorithmic governance. In Proceedings 22nd ACM Conference on ComputerSupported Cooperative Work and Social Computing, , pages 1-35, 2019.
- [19] Abhilash Mishra. AI alignment and social choice: Fundamental limitations and policy implications. arXiv preprint arXiv:2310.16048 , 2023.

- [20] Ritesh Noothigattu, Snehalkumar Gaikwad, Edmond Awad, Sohan Dsouza, Iyad Rahwan, Pradeep Ravikumar, and Ariel Procaccia. A voting-based system for ethical decision making. In Proceedings of the 32th AAAI Conference on Artificial Intelligence , pages 1587-1594, 2018.
- [21] Ritesh Noothigattu, Dominik Peters, and Ariel D Procaccia. Axioms for learning from pairwise comparisons. Advances in Neural Information Processing Systems , 33:17745-17754, 2020.
- [22] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems , 35:27730-27744, 2022.
- [23] Chanwoo Park, Mingyang Liu, Kaiqing Zhang, and Asuman Ozdaglar. Principled RLHF from heterogeneous feedback via personalization and preference aggregation. arXiv preprint arXiv:2405.00254 , 2024.
- [24] R Tyrrell Rockafellar. Convex Analysis . Princeton University Press, 1970.
- [25] Anand Siththaranjan, Cassidy Laidlaw, and Dylan Hadfield-Menell. Distributional preference learning: Understanding and accounting for hidden context in RLHF. arXiv preprint arXiv:2312.08358 , 2023.
- [26] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems , 33:3008-3021, 2020.
- [27] Gokul Swamy, Christoph Dann, Rahul Kidambi, Zhiwei Steven Wu, and Alekh Agarwal. A minimaximalist approach to reinforcement learning from human feedback. arXiv preprint arXiv:2401.04056 , 2024.
- [28] Paolo Viappiani and Craig Boutilier. Optimal Bayesian recommendation sets and myopically optimal choice query sets. Advances in Neural Information Processing Systems , 23, 2010.
- [29] HPeyton Young. Condorcet's theory of voting. The American Political Science Review , 82(4): 1231-1244, 1988.
- [30] Huiying Zhong, Zhun Deng, Weijie J Su, Zhiwei Steven Wu, and Linjun Zhang. Provable multi-party reinforcement learning with diverse human feedback. arXiv preprint arXiv:2403.05006 , 2024.
- [31] Banghua Zhu, Michael Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or k -wise comparisons. In Proceedings of the 40th International Conference on Machine Learning , pages 43037-43067, 2023.
- [32] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 , 2019.

## A Deferred Proofs

## A.1 Proof of Lemma 3.2

Proof. We begin with some observations on g . First, we have that since /lscript is nonnegative, g must also be nonnegative. This along with the fact that lim x →∞ /lscript ( x ) = ∞ , we have that both lim x →∞ g ( x ) = ∞ and lim x →-∞ g ( x ) = ∞ . Together, these imply that L unconstr attains a minimum. Indeed, L unconstr (0 , 0) = 3 g (0) , and there is some bound B such that for all x &gt; B and x &lt; -B , g ( x ) &gt; 3 g (0) . We can therefore restrict the optimization problem to r a , r b ∈ [ -B,B ] without changing the solutions. Since L unconstr is continuous and [ -B,B ] 2 is compact, a minimum is attained.

Next, note that g is convex because compositions of convex functions with monotonic functions and convex combinations of convex functions are convex [24]. From this, we claim that if there is an optimal solution ( r a , r b ) , then ( r a , r a / 2) is also an optimal solution. Indeed, fix such an ( r a , r b ) ,

<!-- formula-not-decoded -->

where the inequality comes from convexity. This implies that ( r a , r a / 2) is also optimal.

By above, we have that if ( r b , r a ) is optimal, it must be the case that r a minimizes

<!-- formula-not-decoded -->

Observe that h is again convex by monotonic composition and convex combinations.

Next, we will make use of the following facts about convex functions. Although they need not be differentiable, right- and left-hand derivatives always exist. For a function k these are defined as

<!-- formula-not-decoded -->

Further, if k is convex, we have that [24]:

- (i) k ′ + and k ′ -are nondecreasing,
- (ii) k ′ -( x ) ≤ k ′ + ( x ) for every x,
- (iii) x minimizes k if and only if k ′ -( x ) ≤ 0 ≤ k ′ + ( x ) ,
- (iv) k ′ + and k ′ -are right and left continuous, respectively .

They also follow standard linearity and chain rule properties, which allow for simpler computation. For example:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(all of these hold for left-hand derivatives by swapping the positions of + and -) [9].

Next, we claim that we can find a valid p (rational with 1 / 2 &lt; p &lt; 1 ) and w &gt; 0 such that g ′ + ( w ) &gt; 0 , while h ′ + ( w ) &lt; 0 . To that end, expanding the first derivative, we have

<!-- formula-not-decoded -->

As long as /lscript ′ -( -x ) + /lscript ′ + ( x ) &gt; 0 , this is strictly more than 0 for p satisfying

<!-- formula-not-decoded -->

For h ,

As long as /lscript ′ -( -x/ 2) + /lscript ′ -( -x ) + /lscript ′ + ( x/ 2) + /lscript ′ + ( x ) &gt; 0 , then this is strictly less than 0 for

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, choose w &gt; 0 such that /lscript ′ -( -w/ 2) &gt; /lscript ′ -( -w ) ≥ 0 . This is possible by using the following procedure. We know /lscript ′ -(0) &gt; 0 (as otherwise /lscript (0) &lt; /lscript ( x ) for all x &lt; 0 , contradicting our assumption on /lscript ). We will split into cases depending on whether /lscript is nondecreasing or strictly-convex (at least one must be true by the theorem assumptions).

First, suppose /lscript is non-decreasing. This implies that /lscript ′ -( x ) ≥ 0 for all x . We know that there is a point x &lt; 0 such that /lscript ′ -( x ) &lt; /lscript ′ -(0) (as otherwise /lscript would eventually become negative). Let d = /lscript ′ -( x ) . Take w such that

<!-- formula-not-decoded -->

Note that /lscript ′ -( -w ) = d because /lscript ′ -is left continuous (from (iv) above). Since -w &lt; 0 , so -w 2 &gt; -w . This implies that /lscript ′ -( -w 2 ) &gt; d , as otherwise -w would not be the supremum of such points. In addition, we have that d ≥ 0 because /lscript is nondecreasing.

Next, suppose /lscript is strictly convex. then /lscript ′ -is strictly increasing. Further, since it is left continuous and /lscript ′ 0 (0) &gt; 0 , there is a γ &gt; 0 such that for all x ∈ [ -γ, 0] , /lscript ′ -( x ) ≥ 0 . Therefore, choosing w = γ will do: /lscript ′ -( -γ ) ≥ 0 by choice of γ , and -γ/ 2 &gt; -γ , so /lscript ′ -( -γ/ 2) &gt; /lscript ′ -( -γ ) since /lscript ′ -is strictly increasing.

For this choice of w , we first claim that the preconditions of denominators being positive hold for (2) and (3). Indeed, let us write z 1 , z 2 , z 3 , z 4 for /lscript ′ -( -2 w ) , /lscript ′ -( -w ) , /lscript ′ + ( w ) , /lscript ′ + (2 w ) . We know that

<!-- formula-not-decoded -->

by properties of convexity, and since /lscript ′ -( -w/ 2) &gt; /lscript ′ -( -w ) ≥ 0 , we have that 0 ≤ z 1 &lt; z 2 . The denominators are of the form z 1 + z 4 and z 1 + z 2 + z 3 + z 4 , which are now both necessarily positive. In addition, we also claim that a rational p with 1 / 2 &lt; p &lt; 1 satisfying both inequalities (2) and (3) will exist. Note that inequality (2) can now be represented as z 4 z 1 + z 4 , and we have that

<!-- formula-not-decoded -->

because 0 ≤ z 1 &lt; z 4 . Additionally, inequality (3) can be represented as

<!-- formula-not-decoded -->

which is at least 1 / 2 because z 3 + z 4 &gt; z 1 + z 2 . Finally, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, there exists some rational p in this interval, which is necessarily between 1 / 2 and 1 , as needed.

To summarize, we have found a valid p and value w &gt; 0 such that h ′ + ( w ) &lt; 0 while g ′ + ( w ) &gt; 0 . Because h ′ + is right continuous (from (iv) above), there is some γ &gt; 0 such that h ′ + ( w + γ ) &lt; 0 as well. We will now show for all ( r a , r b ) ∈ OPT unconstr , r a &gt; w + γ and r b ≤ w . Indeed, note that r a must minimize h , so h ′ + ( r a ) ≥ 0 (from (iii) above), which implies r a &gt; w + c . For r b , suppose which implies that

for a contradiction r b &gt; w as well. Note that g ′ + ( w ) &gt; 0 means g is increasing to the right of w . Let d = min( r a , r b ) -w &gt; 0 . Consider ( r ′ a , r ′ b ) = ( r a -d, r b -d ) . We then have

<!-- formula-not-decoded -->

where the equality holds because r ′ a -r ′ b = r a -r b and the inequality because g is increasing to the right of w . Therefore, we reach a contradiction, because then ( r a , r b ) would not be optimal. Thus, we have found values satisfying the lemma statement with A 1 = w and A 2 = w + γ .

## A.2 Proof of Lemma 3.3

Proof. Fix p inducing a nonempty OPT unconstr satisfying Lemma 3.2 with values A 1 and A 2 . We first claim that for any ( r a , r b ) (regardless of optimality), it is possible to find a θ such that r θ ( a ) = r a and r θ ( b ) = r b . Indeed, note that

<!-- formula-not-decoded -->

Since M = ( 2 1 1 1 ) is invertible with inverse

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, OPT com is nonempty, and is simply the image of OPT unconstr under M -1 . Now, fix ( r a , r b ) ∈ OPT unconstr , and let θ = M -1 ( r a r b ) . By assumption, we have r a &gt; A 2 and r b ≤ A 1 . This implies that θ 1 = r a -r b &gt; A 2 -A 1 , while θ 2 = 2 r b -r a &lt; 2 A 1 -A 2 . Thus, setting A 3 = A 2 -A 1 and A 4 = 2 A 1 -A 2 satisfy the desired properties.

## A.3 Proof of Lemma 3.4

Proof. We first claim that OPT (0) = OPT core . This will follow from showing

<!-- formula-not-decoded -->

Indeed, in L 0 , the copied candidates are in exactly the same location as their counterparts. Hence, each term in L core appears 4 times, one for each combination of original and copy. In addition to these, there are the 6 terms for each ordered pair of ( a, a ′ ) , ( b, b ′ ) , and ( c, c ′ ) . Note that, each r θ ( x ) = r θ ( x ′ ) for each x ∈ { a, b, c } regardless of θ since they are in the same location. Therefore, the /lscript portion is always /lscript (0) , and the corresponding w x /follows x ′ and w x ′ /follows x terms add up to one for each pair. Hence, the total sum of these terms is 3 · /lscript (0) . Since L 0 is equivalent to L core up to positive scaling and translation, they have the same optima.

Finally, fix a θ ∈ OPT (0) . Since θ ∈ OPT core , by Lemma 3.3, θ 1 &gt; A 3 and θ 2 &lt; A 4 . Thus,

<!-- formula-not-decoded -->

by choice of δ . Hence, θ ∈ R c ′ /follows c

## A.4 Proof of Lemma 3.5

Proof. We first show that when optimizing each L ε , it is sufficient to consider only θ coming from a bounded region. Indeed, observe that L ε ( 0 ) = ( 6 2 ) /lscript (0) for all ε . Since lim x →∞ /lscript ( x ) = ∞ , we for any r a , r b , we can simply set

can find some B &gt; 0 such that for all x &gt; B , /lscript ( x ) &gt; ( 6 2 ) /lscript (0) 1 -p = L ε ( 0 ) 1 -p . For a pair of candidates x = y ∈ C com , in the two terms concerning these candidates, we have

<!-- formula-not-decoded -->

Applying this to { a, b } and { b, c } ,

<!-- formula-not-decoded -->

This implies that we may restrict our attention to θ in the region

<!-- formula-not-decoded -->

Indeed, for θ / ∈ R bounded , either | θ 1 | &gt; B or | θ 1 + θ 2 | &gt; B . In either case, we have L ε ( θ ) ≥ (1 -p ) /lscript ( B ) &gt; L ε ( 0 ) .

Note that L ε ( θ ) is continuous not only in θ , but also in ε . Additionally, R bounded is closed and bounded, and hence, compact. Therefore, by Berge's Maximum Theorem, OPT ( ε ) is nonempty and upper semi-continuous in ε [3]. As per the definition of upper semi-continuous, since OPT (0) ⊆ R c ′ /follows c , an open set, for sufficiently small ε &gt; 0 , OPT ( ε ) ⊆ R c ′ /follows c . Finally, note that R c ′ /follows c ∩ R bounded is compact, so a minimum is attained, and this minimum must therefore be strictly larger than the values attained by members of OPT ( ε ) .

## A.5 Proof of Theorem 3.6

Proof. Without loss of generality, we may assume that inf x /lscript ( x ) = 0 , as otherwise we could translate /lscript without affecting the optimization problem. Fix a profile π with feasible PMC ranking σ , and let θ PMC be a non-degenerate parameter vector that induces σ .

First, we show that inf θ L maj ( θ ; π, /lscript ) = 0 . Indeed, note that c · θ PMC ∈ R π for all c &gt; 0 . Further, note that for any a, b with w a /follows b ( π ) &gt; 1 / 2 , r θ PMC ( b ) -r θ PMC ( a ) &lt; 0 . Therefore, by making c large, the nonzero terms in L maj will have an input to /lscript negative and becoming arbitrarily large in magnitude. Since /lscript is nondecreasing, these approach the infemum of 0 .

/negationslash

Next, for any σ ′ = σ , inf θ L maj ( θ ; π, /lscript ) ≥ /lscript (0) . Indeed, there must be some pair of candidates a, b with a /follows σ b and b /follows σ ′ a . For any θ ∈ R σ ′ , r θ ( b ) ≥ r θ ( a ) , so /lscript ( r θ ( b ) -r θ ( a )) ≥ /lscript (0) , and this lower bounds the loss function.

## A.6 Proof of Theorem 3.7

Proof. We construct an explicit instance and pairwise majority relationships such that no matter what feasible ranking a rule picks, there is an underlying profile where that output was a PO violation.

We will have 9 candidates; 8 will be labeled c + i and c -i for i = 1 , 2 , 3 , 4 , and one labeled c ∗ . They will have feature vectors in R 4 . Each c ± i will be located at x c ± i = ± e i where e i is the i 'th standard basis vector, i.e., c + 2 is at (0 , 1 , 0 , 0) and c -4 is at (0 , 0 , 0 , -1) . Finally, c ∗ will be located at (1 / 5 , 1 / 5 , 1 / 5 , 1 / 5) .

There will be 5 voters. Their pairwise majority graph will be as follows. Candidate c ∗ will pairwise beat all others. In addition, each c + i will pairwise beat each c -j . Among the c + i candidates, there will be a cycle c + 1 /follows c + 2 /follows c + 3 /follows c + 4 /follows c + 1 and between the remaining two pairs c + 1 /follows c + 3 and c + 2 /follows c + 4 . The c -i candidates will be the exact reverse of this, i.e., a cycle c -4 /follows c -3 /follows c -2 /follows c -1 /follows c -4 , along with c -3 /follows c -1 and c -4 /follows c -2 . A pictorial representation can be found in Figure 1.

A C1 rule must pick a θ solely based on the pairwise majority graph. We will show that regardless of what θ it outputs, this will lead to a PO violation.

/negationslash

Figure 1: Graph showing pairwise majority relationship between candidates. Regular edges show relationships among c + i candidates and among c -i candidates. Thick edges indicate that c ∗ pairwise beats all candidates, and each c + i pairwise beats each c -j candidate.

![Image](data:image/png;base64,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)

To that end, the first fact we will show is that no θ can rank c ∗ first. Indeed, for any θ , r θ ( c ∗ ) = 1 5 ∑ i θ i ≤ 1 5 ∑ i | θ i | &lt; max i | θ i | . On the other hand, for i maximizing | θ i | , at least one of c ± i will achieve this reward, strictly larger than r θ ( c ∗ ) . Hence, regardless of the output θ , some candidate must be ranked above c ∗ . We will show that this leads to a PO violation.

To construct profiles consistent with the pairwise majority graph, voters will always have rankings of the following form:

<!-- formula-not-decoded -->

for some { i, j, k, /lscript } = { 1 , 2 , 3 , 4 } . In other words, they will rank a single + candidate above c ∗ and the rest in some order, followed by all -candidates in the reverse order. This is always achievable with the voter vector that puts values 1 , 3 ε, 2 ε, 1 ε in entires i, j, k , and /lscript , respectively, for some small ε &gt; 0 .

/negationslash

Fix an output θ with induced ranking σ . There must be at least one candidate c = c ∗ ranked above c ∗ . We now split into cases depending on which candidate this is. For each choice, we will construct a profile consistent with the pairwise majority graph where the candidate above c ∗ is Pareto dominated by c ∗ . We will describe each voter's ranking only by an ordering over the + candidates, assuming they otherwise take the form described in (4). Note that c ∗ is always ranked second, so if a candidate is never ranked first, they are Pareto dominated by c ∗ . The profiles for each candidate c + i can be found in the following table. One can check that all pairwise relationships are satisfied, and the corresponding c + i is never ranked first.

Table 1: Profiles with 5 voters and consistent with the pairwise majority graph where where the corresponding candidate is PO-dominated by c ∗ . The notation 1 : (2 , 1 , 3 , 4) implies one voter has the ranking in the form of (4) with ( i, j, k, /lscript ) = (2 , 1 , 3 , 4) .

| c + 1                                                                       | c + 2                                                    | c + 3                                                    | c + 4                                                    |
|-----------------------------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
| 1: (2 , 1 , 3 , 4) 1: (3 , 1 , 2 , 4) 1: (3 , 2 , 4 , 1) 2: (4 , 1 , 2 , 3) | 2: (1 , 2 , 3 , 4) 2: (3 , 2 , 4 , 1) 1: (4 , 1 , 2 , 3) | 2: (1 , 2 , 3 , 4) 2: (2 , 3 , 4 , 1) 1: (4 , 1 , 2 , 3) | 2: (1 , 2 , 3 , 4) 2: (2 , 4 , 1 , 3) 1: (3 , 4 , 1 , 2) |

Finally, if a -candidate is ranked above c ∗ , then any of the following profiles work, as all -candidates are Pareto dominated by c ∗ with rankings shown in (4).

## B PMC Infeasibility Example

Consider the case with d = 3 and seven candidates: one special candidate a ∗ located at (1 / 4 , 1 / 4 , 1 / 4) , and six others c ± i located at standard basis vectors e ± i . We have three voters with parameter vectors (1 , 2 ε, ε ) , (2 ε, 1 , ε ) , and (2 ε, ε, 1) , where ε &lt; 1 / 5 is a small positive number. These voters have the following induced rankings:

| Rank          | v 1                                     | v 2                                     | v 3                                     |
|---------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| 1 2 3 4 5 6 7 | c + 1 a ∗ c + 2 c + 3 c - 3 c - 2 c - 1 | c + 2 a ∗ c + 1 c + 3 c - 3 c - 1 c - 2 | c + 3 a ∗ c + 1 c + 2 c - 2 c - 1 c - 3 |

We argue the ranking a ∗ /follows c + 1 /follows c + 2 /follows c + 3 /follows c -3 /follows c -2 /follows c -1 is a PMC ranking but no linear reward function can position a ∗ at the top of this ranking.

For any reward vector θ = ( θ 1 , θ 2 , θ 3 ) ∈ R 3 , the reward for a ∗ is :

<!-- formula-not-decoded -->

Given the placement of c ± i at the standard basis vectors, each c ± i achieves a reward equivalent to one of the absolute values of the components of θ , thus surpassing r θ a ∗ since

<!-- formula-not-decoded -->

## C Additional Axiomatic Properties of Social Choice-Based Rules

We begin by stating another prominent axiom from social choice theory.

Definition C.1 (Separability) . A ranking aggregation rule satisfies ranking separability (or separability for short) if, when two profiles yield identical output rankings, when combined into a single profile, this should also produce the same output ranking.

Ranking separability preserves consistency in aggregation outputs and ensures stable decisions across similar preference distributions.

## C.1 Copeland Violates Separability

Theorem C.2. Both Copeland (in traditional social choice) and LCPO (in linear social choice) fail separability.

Proof. Consider the following two profiles on 7 candidates with five and three voters each:

|   Rank | v 1   | v 2   | v 3   | v 4   | v 5   |   Rank | v 6   | v 7   | v 8   |
|--------|-------|-------|-------|-------|-------|--------|-------|-------|-------|
|      1 | a     | b     | b     | c     | d     |      1 | a     | b     | c     |
|      2 | g     | a     | a     | e     | c     |      2 | d     | a     | a     |
|      3 | d     | c     | e     | f     | f     |      3 | b     | e     | e     |
|      4 | e     | e     | d     | g     | g     |      4 | f     | d     | b     |
|      5 | f     | d     | f     | b     | b     |      5 | c     | c     | f     |
|      6 | b     | g     | g     | a     | e     |      6 | g     | g     | g     |
|      7 | c     | f     | c     | d     | a     |      7 | e     | f     | d     |

In the first profile, the Copeland scores are 5 , 4 , 3 , 3 , 3 , 2 , 1 for candidates a, b, c, d, e, f, g , respectively. Similarly, in the second profile, they are 6 , 5 , 3 , 3 , 3 , 1 , 0 . So under any consistent tie-breaking rule, both of these profiles would output the ranking a /follows b /follows c /follows d /follows e /follows f /follows g .

However, if we combine these two profiles, then the score of a is 5 while the score of b is 6 , and thus, b will be ranked above a , violating separability.

To see that this also holds for LCPO, note that when every ranking is feasible, LCPO coincides with Copeland. We can simply have 7 candidates in R 7 all located at unit vectors, and voters with inputs from the above profiles.

## C.2 Linear Kemeny Rule

Next, we consider a different rule from social choice theory, the Kemeny rule, which can be transformed to the linear setting while maintaining the separability can be achieved along with PMC. Given an input profile π , the Kemeny rule returns a ranking σ ∗ that minimizes the total number of pairwise disagreements with voters rankings, i.e.

<!-- formula-not-decoded -->

where S m contains all possible permutations of the m candidates. This expression can be equivalently written as

/negationslash

<!-- formula-not-decoded -->

Here, we define the linear Kemeny rule , which outputs a parameter vector θ ∗ that induces a ranking that minimizes the total number of pairwise disagreements with voters' rankings, i.e.,

/negationslash

<!-- formula-not-decoded -->

Note that this rule conforms to the standard loss formulation, where the loss function is binary: it is 0 if two rankings agree with respect to the relative ranking of a pair of candidates and 1 otherwise. Since binary loss is not convex, it does not fit in the impossibility result of Theorem 3.1.

Note that Kemeny is generally NP-hard to compute; however, even for linear Kemeny, there is at least an exponential time aglorithm by brute-force computing the score of every ranking, and determining whether or not it is feasible.

Theorem C.3. Linear Kemeny satisfies PMC and separability.

Proof. PMC holds because the linear Kemeny score minimizes disagreements even among nontransitive rankings, making the PMC ranking the optimal choice whenever it is feasible. Separability is evident as the Kemeny score of a ranking over two datasets is simply the sum of the scores in each dataset. If the same ranking minimizes the score in both datasets independently, it will also minimize the score in their combination.

Theorem C.4. Linear Kemeny does not satisfy PO or majority consistency.

Proof. Consider the scenario with 20 candidates whose feature vectors are represented in the table below:

|   Candidate | Feature Vector                        |
|-------------|---------------------------------------|
|           1 | (2000000 , 0 , 0 , 0 , 0 , 0 , 0)     |
|           2 | (0 , 2000000 , 0 , 0 , 0 , 0 , 0)     |
|           3 | (0 , 200000 , 0 , 0 , 0 , 0 , 0)      |
|           4 | (0 , 100000 , 100000 , 0 , 0 , 0 , 0) |
|           5 | (0 , 0 , 200000 , 0 , 0 , 0 , 0)      |
|           6 | (0 , 0 , 20000 , 0 , 0 , 0 , 0)       |
|           7 | (0 , 0 , 10000 , 10000 , 0 , 0 , 0)   |
|           8 | (0 , 0 , 0 , 20000 , 0 , 0 , 0)       |
|           9 | (0 , 0 , 0 , 2000 , 0 , 0 , 0)        |
|          10 | (0 , 0 , 0 , 1000 , 1000 , 0 , 0)     |
|          11 | (0 , 0 , 0 , 0 , 2000 , 0 , 0)        |
|          12 | (0 , 0 , 0 , 0 , 200 , 0 , 0)         |
|          13 | (0 , 0 , 0 , 0 , 100 , 100 , 0)       |
|          14 | (0 , 0 , 0 , 0 , 0 , 200 , 0)         |
|          15 | (0 , 0 , 0 , 0 , 0 , 20 , 0)          |
|          16 | (0 , 0 , 0 , 0 , 0 , 10 , 10)         |
|          17 | (0 , 0 , 0 , 0 , 0 , 0 , 20)          |
|          18 | (0 , 0 , 0 , 0 , 0 , 0 , 2)           |
|          19 | (1 , 0 , 0 , 0 , 0 , 0 , 1)           |
|          20 | (2 , 0 , 0 , 0 , 0 , 0 , 0)           |

Each vector is constructed such that candidates are prioritized based on the magnitude of their first non-zero entry, leading to a natural ordering within grouped subsets: { 1 , 2 } /follows { 3 , 4 , 5 } /follows { 6 , 7 , 8 } /follows { 9 , 10 , 11 } /follows { 12 , 13 , 14 } /follows { 15 , 16 , 17 } /follows { 18 , 19 , 20 } . We will have six voters, with rankings induced by the parameter vectors described below.

| Voter                   | Parameter Vector                                                                                                                                                        |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v 1 v 2 v 3 v 4 v 5 v 6 | (2 , 1 , 7 , 6 , 5 , 4 , 3) (3 , 2 , 1 , 7 , 6 , 5 , 4) (4 , 3 , 2 , 1 , 7 , 6 , 5) (5 , 4 , 3 , 2 , 1 , 7 , 6) (6 , 5 , 4 , 3 , 2 , 1 , 7) (7 , 6 , 5 , 4 , 3 , 2 , 1) |

Each voter ranks candidates within the group in increasing order except for a single reversed group. For instance, v 1 ranks candidates as 1 /follows 2 /follows 5 /follows 4 /follows 3 and so forth. Under this setup, no parameter vector θ can create a ranking that satisfies all voters' preferences due to the cyclic nature and individual group preferences.

We can check that linear Kemeny rule outputs a ranking in which candidate 2 is ranked above 1. From this analysis, we also conclude that the rules does not satisfy majority consistency either.

A possibly easy fix to the problem that linear Kemeny does not satisfy PO would be to enforce the PO criterion, as we did for the LCPO, i.e., to restrict to parameter vectors that respect PO. However, we show that if we do that, then linear Kemeny subject to PO does not satisfy separability anymore.

Theorem C.5. Linear Kemeny subject to PO violates separability and majority consistency.

Proof. First, consider a set of candidates and a profile similar to the one that is given in the proof of Theorem C.4. When we restrict to reward functions that 1 above 2 , then we can check that Linear Kemeny subject to PO outputs one of the rankings that are in the input profile. Without loss of generality, assume that it outputs the ranking of v 1 .

Second, consider the same set of candidates and three voters, with rankings induced by the following parameter vectors:

| Voter             | Parameter Vector                                                                    |
|-------------------|-------------------------------------------------------------------------------------|
| v ′ 1 v ′ 2 v ′ 3 | (2 , 1 , 7 , 6 , 5 , 4 , 3) (2 , 1 , 7 , 6 , 5 , 4 , 3) (1 , 7 , 6 , 5 , 4 , 3 , 2) |

In this case, Linear Kemeny subject to PO outputs the ranking of v ′ 1 which is the same with this of voter v 1 .

When the two profiles are combined, then we do not anymore restrict on rankings in which 1 is above 2 , since 1 does not Pareto dominates 2 anymore. Then, we can check that linear Kemeny subject to PO outputs a different ranking than before in which 2 is ranked above 1 . From this example, we see that linear Kemenery subject to PO still violates majority consistency.

## C.3 Leximax Plurality

Plurality is probably the most ubiquitous voting rule in the world. Its ranking variant ranks the candidates in decreasing order with respect to their plurality scores . The plurality score of a candidate is equal to the number of her appearances in the first position. This rule is known to satisfy several axioms but in linear social choice cannot be directly applied, as not all rankings are feasibly.

Similarly to leximax Copeland, we define Leximax Plurality as follows. It ranks first the candidate with the highest plurality score that can be ranked first under some parameter vector. Subject to this first position, it ranks second the candidate with the highest plurality score that can feasibly be ranked second, and so on, until all the positions are filled.

Theorem C.6. Leximax Plurality satisfies majority consistency, winner monotonicity and separability.

Proof. Note that leximax Plurality always returns a ranking in which the candidate with the highest plurality score is ranked first, since there exists at least one feasible ranking in which this candidate is ranked first. From this observation, we immediately see that leximax Plurality satisfies majority consistency.

Now, suppose that on input profile π , the rule outputs a ranking such that candidate a is ranked first. This means that a has the highest plurality score. Now, consider a profile π ′ which is similar to π with the only exception being a ranking in which a is ranked in a higher position. It is clear that a continues to have the highest plurality score and therefore leximax Plurality will output a ranking in which a is ranked first. Therefore, winner monotonicity is satisfied. .

It remains to prove that the rule satisfies separability. Suppose that under two different profiles π 1 and π 2 , the rule outputs σ , and under the aggregated profile π 3 , it outputs σ ′ . We will show that σ = σ ′ . One main observation is that if a candidate a has a higher plurality score than a candidate b under both π 1 and π 2 , then a has a higher plurality score than b under π 3 as well. We will show the desired property by induction on the positions of the ranking σ . Start from the first position in which say candidate a is ranked first. From above, we know that a has the highest plurality score under both π 1 and π 2 , which remains true in π 3 , and therefore a is ranked first in σ ′ . Now, assume that up to position t -1 , σ and σ ′ are similar and denote with a ′ the candidate that is ranked at position t of σ . We denote by S 1 , S 2 and S 3 the set of candidates that are not ranked among the first t positions in σ and have higher plurality score than a ′ under π 1 , π 2 , and π 3 respectively. Since, a ′ is ranked at the t -th postion in σ , we get that, subject to the fixed first t -1 position, no candidate in S 1 ∪ S 2 can be ranked at the t -th position. The theorem follows by noticing that S 3 ⊆ S 1 ∪ S 2 , which follows from the main observation above.