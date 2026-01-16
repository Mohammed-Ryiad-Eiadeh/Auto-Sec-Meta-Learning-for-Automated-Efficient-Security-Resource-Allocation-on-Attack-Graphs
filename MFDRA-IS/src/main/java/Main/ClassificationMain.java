package Main;

import StoreRetrieveHashmap.org.RetrieveResourceAllocationAndTopAllocationAsHashMap;
import org.apache.commons.math3.stat.inference.WilcoxonSignedRankTest;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.liblinear.LibLinearClassificationTrainer;
import org.tribuo.classification.mnb.MultinomialNaiveBayesTrainer;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.kernel.KernelSVMTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.common.nearest.KNNClassifierOptions;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.interop.tensorflow.*;
import org.tribuo.interop.tensorflow.example.CNNExamples;
import org.tribuo.interop.tensorflow.example.MLPExamples;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.optimisers.AdaGradRDA;
import org.tribuo.transform.TransformTrainer;
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.transformations.LinearScalingTransformation;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class ClassificationMain implements RetrieveResourceAllocationAndTopAllocationAsHashMap {
    public static void main(String[] args) throws IOException {
        // create row processor to handle the header of the data
        var rowProcessor = GetRowProcessor();

        // read the training and testing dataset
        var trainData = new CSVDataSource<>(Paths.get(System.getProperty("user.dir"), "\\Train Embeddings RW.csv"), rowProcessor, true);
        var testData = new CSVDataSource<>(Paths.get(System.getProperty("user.dir"), "\\Test Embeddings RW.csv"), rowProcessor, true);

        // generate Tribute form data portions of the read dataset
        var train = new MutableDataset<>(trainData);
        var test = new MutableDataset<>(testData);

        // show the training data features like size
        System.out.printf("train data size = %d, number of features = %d, number of labels = %d%n", train.size(),
                train.getFeatureMap().size(),
                train.getOutputInfo().size());

        // show the testing data features like size
        System.out.printf("test data size = %d, number of features = %d, number of labels = %d%n", test.size(),
                test.getFeatureMap().size(),
                test.getOutputInfo().size());

        // define label evaluation for evaluation purposes
        LabelEvaluation Evaluator;

        // define the map of top resource allocation approaches
        var map = RetrieveResourceAllocationAndTopAllocationAsHashMap
                .getTopAllocationApproaches("Top Allocation Methods");

        //
        var performanceMatrix = RetrieveResourceAllocationAndTopAllocationAsHashMap
                .getPerformanceMatrixAsHashMap(Path.of(System.getProperty("user.dir") + "\\Allocation Results.csv"));

        // define the k value involved in hit-at-k
        var k = 10;

        // define two variable to store the start and end date
        var sTime = 0L;
        var eTime = 0L;

        var mlpGraph0 = new ResMlp((long) train.getFeatureMap().size(), (long) train.getOutputInfo().size(), 250L);
        var mlpOptimizer0 = GradientOptimiser.GRADIENT_DESCENT;
        var mlpOptimizerParameters0 = Map.of("learningRate", 0.01f);
        var TF_Trainer_MLP0 = new TensorFlowTrainer<>(mlpGraph0.getModel().graphDef,
                mlpGraph0.getModel().outputName,
                mlpOptimizer0,
                mlpOptimizerParameters0,
                new DenseFeatureConverter(mlpGraph0.getModel().inputName),
                new LabelConverter(),
                16,
                100,
                16,
                -1);
        var TF_MLP_Learner0 = TF_Trainer_MLP0.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(TF_MLP_Learner0, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------MLP Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, TF_MLP_Learner0, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, TF_MLP_Learner0, map, performanceMatrix, k).get(1));

        System.exit(0);

        // define, train, and test LeNet-5
        var inputName = "LeNet5";
        var LeNet_5 = CNNExamples.buildLeNetGraph(inputName, 32, 255, train.getOutputs().size());
        var optimizer = GradientOptimiser.ADAGRAD;
        var optimizerParameters = Map.of("learningRate", 0.01f, "initialAccumulatorValue", 0.01f);
        var TF_Trainer = new TensorFlowTrainer<>(LeNet_5.graphDef,
                LeNet_5.outputName,
                optimizer,
                optimizerParameters,
                new ImageConverter(inputName, 32, 32, 1),
                new LabelConverter(),
                16,
                20,
                16,
                -1);
        var TF_Learner = TF_Trainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(TF_Learner, test); //degreeCentralityWithMarkovBlanket by degreeCentralityWithInDegreeNodes
        eTime = System.currentTimeMillis();
        System.out.println("----------------LeNet-5 Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, TF_Learner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, TF_Learner, map, performanceMatrix, k).get(1));
        System.out.println("The Std is : " + hitAt_K(test, TF_Learner, map, performanceMatrix, k).get(2));

        var inName = "MLP";
        var mlpGraph = MLPExamples.buildMLPGraph(inName, train.getFeatureMap().size(), new int[] {300, 200, 30}, train.getOutputs().size());
        var mlpOptimizer = GradientOptimiser.ADAGRAD;
        var mlpOptimizerParameters = Map.of("learningRate", 0.01f, "initialAccumulatorValue", 0.01f);
        var TF_Trainer_MLP = new TensorFlowTrainer<>(mlpGraph.graphDef,
                mlpGraph.outputName,
                mlpOptimizer,
                mlpOptimizerParameters,
                new DenseFeatureConverter(inName),
                new LabelConverter(),
                25,
                100,
                25,
                -1);
        var TF_MLP_Learner = TF_Trainer_MLP.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(TF_MLP_Learner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------MLP Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, TF_MLP_Learner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, TF_MLP_Learner, map, performanceMatrix, k).get(1));

        // define, train, and test factorization machine classifier
        var FactorizationMachine = new FMClassificationTrainer(new Hinge(), new AdaGradRDA(0.1, 0.8), 1, 1000, 1, Trainer.DEFAULT_SEED, 10, 0.2);
        var FMLearner = FactorizationMachine.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(FMLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------Factorization Machine Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, FMLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, FMLearner, map, performanceMatrix, k).get(1));
        System.out.println("The Std is : " + hitAt_K(test, FMLearner, map, performanceMatrix, k).get(2));
                // define, train, and test SVM classifier
        var SVMTrainer = new KernelSVMTrainer(new Linear(), 0.1, 1, Trainer.DEFAULT_SEED);
        var SVMLearner = SVMTrainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(SVMLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------Support Vector Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, SVMLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, SVMLearner, map, performanceMatrix, k).get(1));
        System.out.println("The Std is : " + hitAt_K(test, SVMLearner, map, performanceMatrix, k).get(2));

        // define, train, and test LR classifier
        var LRTrainer = new LogisticRegressionTrainer();
        var LRLearner = LRTrainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(LRLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------Logistic Regression Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, LRLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, LRLearner, map, performanceMatrix, k).get(1));

        // define, train, and test KNN classifier
        var KNNTrainer = new KNNClassifierOptions();
        KNNTrainer.knnK = 3;
        KNNTrainer.distType = DistanceType.COSINE;
        var KNNLearner = KNNTrainer.getTrainer().train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(KNNLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------KNN Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, KNNLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, KNNLearner, map, performanceMatrix, k).get(1));

                // define, train, and test CART classifier
        var CartTrainer = new CARTClassificationTrainer();
        var CartLearner = CartTrainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(CartLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------CART Tree Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, CartLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, CartLearner, map, performanceMatrix, k).get(1));

        // define, train, and test LibLinear classifier
        var LibLinTrainer = new LibLinearClassificationTrainer();
        var LibLinLearner = LibLinTrainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(LibLinLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------LibLinear Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, LibLinLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, LibLinLearner, map, performanceMatrix, k).get(1));
        System.out.println("The Std is : " + hitAt_K(test, LibLinLearner, map, performanceMatrix, k).get(2));

                // define, train, and test XGBoost classifier
        var XGBTrainer = new XGBoostClassificationTrainer(10);
        var XGBLearner = XGBTrainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(XGBLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------XGBoost Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, XGBLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, XGBLearner, map, performanceMatrix, k).get(1));
        System.out.println("The Std is : " + hitAt_K(test, XGBLearner, map, performanceMatrix, k).get(2));

        // define, train, and test MNB classifier
        var MNBTrainer = new MultinomialNaiveBayesTrainer();
        var transformation = new TransformationMap(List.of(new LinearScalingTransformation(0, 1)));
        var transformedTrainer = new TransformTrainer<>(MNBTrainer, transformation);
        var MNBLearner = transformedTrainer.train(train);
        sTime = System.currentTimeMillis();
        Evaluator = new LabelEvaluator().evaluate(MNBLearner, test);
        eTime = System.currentTimeMillis();
        System.out.println("----------------MNB Performance\n" + Evaluator +
                "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() +
                "\n" + "Duration Time is : " + Util.formatDuration(sTime, eTime) + "\n\n");
        System.out.println("The acc based on the hit at K is : " + hitAt_K(test, MNBLearner, map, performanceMatrix, k).get(0));
        System.out.println("The average performance is : " + hitAt_K(test, MNBLearner, map, performanceMatrix, k).get(1));
    }

    /**
     * this method is used to construct Row Processor to process the data header
     * @return the constructed row processor
     */
    static RowProcessor<Label> GetRowProcessor() {
        // creat an array list to process the graph id field
        var fieldProcessor = new ArrayList<FieldProcessor>();
        fieldProcessor.add(new IdentityProcessor("GraphID"));

        // creat hashmap to hold the fields name of the dataset
        var FeatureProcessor = new HashMap<String, FieldProcessor>();
        FeatureProcessor.put("AT.*", new DoubleFieldProcessor("AT.*"));

        // creat class label processor for the classes in the dataset to construct the label factory
        var ClassProcessor = new FieldResponseProcessor<>("Class", "nan", new LabelFactory());

        // return the row processor of the generated labels
        return new RowProcessor
                .Builder<Label>()
                .setRegexMappingProcessors(FeatureProcessor)
                .setFieldProcessors(fieldProcessor)
                .build(ClassProcessor);
    }

    static List<Double> hitAt_K(MutableDataset<Label> testPart, Model<Label> trainedModel, HashMap<String, ArrayList<String>> map, HashMap<String, List<String>> performanceMatrix, int top_K) {
        if (map.isEmpty()) {
            throw new IllegalArgumentException("The map holding the graph with their top-allocation approaches is empty");
        }
        double accuracy = 0.0;
        double avgAllocationPerformance = 0d;
        double[] costRelativeReductionVal = new double[testPart.size()];
        int counter = 0;
        for (Example<Label> sample : testPart.getData()) {
            String[] instanceWords = sample.
                    toString().
                    replace("(", " ").
                    replace(",", " ").
                    split(" ");

            String graphID = " ";
            for (String word : instanceWords) {
                if (word.startsWith("GraphID")) {
                    graphID = word.split("@")[1];
                }
            }
            String prediction = trainedModel.predict(sample).getOutput().getLabel();
            ArrayList<String> top_K_GroundTruths = new ArrayList<>();
            for (int i = 0; i < top_K; i++) {
                top_K_GroundTruths.add(map.get(graphID).get(i).toUpperCase(Locale.ROOT));
            }
            if (top_K_GroundTruths.contains(prediction.toUpperCase(Locale.ROOT))) {
                accuracy++;
            }
            var index = performanceMatrix.get("Allocation").stream().map(String::toUpperCase).toList().indexOf(prediction.toUpperCase());
            var costRelativeReduction = performanceMatrix.get(graphID).get(index);
            costRelativeReductionVal[counter] = Double.parseDouble(costRelativeReduction);
            counter++;
            avgAllocationPerformance += Double.parseDouble(costRelativeReduction);
        }
        // This variable will hold the sum of squared deviations from the mean (variance calculation).
        double innerPart = 0;
        // Iterate over each value in the costRelativeReductionVal (CRVal represents each data point)
        for (double CRVal : costRelativeReductionVal) {
            innerPart += Math.pow((CRVal - avgAllocationPerformance / testPart.size()), 2); // Squared deviation from the mean
        }
        // Calculate the standard deviation (for population std, use size of the data set)
        double Std = Math.sqrt(innerPart / (costRelativeReductionVal.length));
        System.out.println(Arrays.toString(costRelativeReductionVal));
        return List.of(accuracy / testPart.getData().size(), avgAllocationPerformance / testPart.getData().size(), Std);
    }
}
