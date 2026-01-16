package Main;

import org.tensorflow.Graph;
import org.tensorflow.framework.initializers.Glorot;
import org.tensorflow.framework.initializers.VarianceScaling;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tribuo.Trainer;
import org.tribuo.interop.tensorflow.example.GraphDefTuple;

public class ResMlp {
    private final Long numberOfFeatures;
    private final Long numberOfOutputLabels;
    private final Long hiddenDimension;

    public ResMlp(Long numberOfFeatures, Long numberOfOutputLabels, Long hiddenDimension) {
        this.numberOfFeatures = numberOfFeatures;
        this.numberOfOutputLabels = numberOfOutputLabels;
        this.hiddenDimension = hiddenDimension;
    }

    public GraphDefTuple getModel() {
        // This object is used to write operations into the graph
        Graph resMlpGraph = new Graph();
        Ops resMlpOps = Ops.create(resMlpGraph);
        Glorot<TFloat32> resMlpGraphInitializer = new Glorot<>(VarianceScaling.Distribution.TRUNCATED_NORMAL,
                Trainer.DEFAULT_SEED);

        // The input block thar we use to feed the features into
        String resMlpInput = "RES_MLP_INPUT";
        Placeholder<TFloat32> inputLayer = resMlpOps
                .withName(resMlpInput)
                .placeholder(TFloat32.class,
                        Placeholder.shape(Shape.of(-1, numberOfFeatures)));

        // Input projection (num of features -> hidden dimension)
        Variable<TFloat32> projWeights = resMlpOps
                .variable(resMlpGraphInitializer.
                        call(resMlpOps,resMlpOps.array(numberOfFeatures, hiddenDimension), TFloat32.class));
        Variable<TFloat32> projBiases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> projActivationOutput = resMlpOps.nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(inputLayer, projWeights), projBiases));
        /* Residual Block 1
         * With Skip Connection
         * Residual Block 1: h1 = h0 + MLP(h0) */

        // Block 1 - FC1; (hiddenDimension -> hiddenDimension)
        Variable<TFloat32> fc1Weights = resMlpOps
                .variable(resMlpGraphInitializer.
                        call(resMlpOps,resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc1Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc1Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(projActivationOutput, fc1Weights), fc1Biases));

        // Block 1 - FC2; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc2Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc2Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc2Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(fc1Activation, fc2Weights), fc2Biases));

        // Residual connection: h1 = h0 + f(h0)
        Add<TFloat32> blook1Output = resMlpOps.math.add(projActivationOutput, fc2Activation);

        /* Residual Block 2
         * With Skip Connection
         * Residual Block 2: h2 = h1 + MLP(h1) */

        // Block 2 - FC3; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc3Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc3Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc3Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(blook1Output, fc3Weights), fc3Biases));

        // Block 2 - FC4; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc4Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc4Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc4Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(fc3Activation, fc4Weights), fc4Biases));

        // Residual connection: h2 = h1 + f(h1)
        Add<TFloat32> blook2Output = resMlpOps.math.add(blook1Output, fc4Activation);

        /* Residual Block 3
         * With Skip Connection
         * Residual Block 3: h3 = h2 + MLP(h2) *//*

        // Block 3 - FC5; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc5Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc5Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc5Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(blook2Output, fc5Weights), fc5Biases));

        // Block 3 - FC6; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc6Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc6Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc6Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(fc5Activation, fc6Weights), fc6Biases));

        // Residual connection: h3 = h2 + f(h2)
        Add<TFloat32> blook3Output = resMlpOps.math.add(blook2Output, fc6Activation);

        *//* Residual Block 4
         * With Skip Connection
         * Residual Block 4: h4 = h3 + MLP(h3) *//*

        // Block 4 - FC7; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc7Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc7Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc7Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(blook3Output, fc7Weights), fc7Biases));

        // Block 4 - FC8; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc8Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc8Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc8Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(fc7Activation, fc8Weights), fc8Biases));

        // Residual connection: h4 = h3 + f(h3)
        Add<TFloat32> blook4Output = resMlpOps.math.add(blook3Output, fc8Activation);*/

        /* Residual Block 5
         * With Skip Connection
         * Residual Block 5: h5 = h4 + MLP(h4) *//*

        // Block 5 - FC9; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc9Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc9Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc9Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(blook4Output, fc9Weights), fc9Biases));

        // Block 5 - FC10; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc10Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc10Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc10Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(fc9Activation, fc10Weights), fc10Biases));

        // Residual connection: h5 = h4 + f(h4)
        Add<TFloat32> blook5Output = resMlpOps.math.add(blook4Output, fc10Activation);

        *//* Residual Block 6
         * With Skip Connection
         * Residual Block 6: h6 = h5 + MLP(h5) *//*

        // Block 6 - FC11; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc11Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc11Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc11Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(blook5Output, fc11Weights), fc11Biases));

        // Block 6 - FC12; (hiddenDimension -> hiddenDimension)
        Variable <TFloat32> fc12Weights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, hiddenDimension), TFloat32.class));
        Variable<TFloat32> fc12Biases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(hiddenDimension), resMlpOps.constant(0.1f)));
        Relu<TFloat32> fc12Activation = resMlpOps
                .nn.relu(resMlpOps.math.add(resMlpOps.linalg.matMul(fc11Activation, fc12Weights), fc12Biases));

        // Residual connection: h6 = h5 + f(h5)
        Add<TFloat32> blook6Output = resMlpOps.math.add(blook5Output, fc12Activation);*/

        // Output Layer (hiddenDimension -> numberOfOutputLabels)
        Variable<TFloat32> outputWeights = resMlpOps
                .variable(resMlpGraphInitializer
                        .call(resMlpOps, resMlpOps.array(hiddenDimension, numberOfOutputLabels), TFloat32.class));
        Variable<TFloat32> outputBiases = resMlpOps
                .variable(resMlpOps.fill(resMlpOps.array(numberOfOutputLabels), resMlpOps.constant(0.1f)));
        Add<TFloat32> output = resMlpOps
                .math.add(resMlpOps.linalg.matMul(blook2Output, outputWeights), outputBiases);


        // Extract the graph def and op name
        GraphDef graphDef = resMlpGraph.toGraphDef();
        String outputName = output.op().name();

        return new GraphDefTuple(graphDef, resMlpInput, outputName);
    }
}
