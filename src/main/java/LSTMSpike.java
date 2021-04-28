import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Zeros;
import org.tensorflow.op.nn.BlockLSTM;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import util.TensorValues;

import java.util.Arrays;

public class LSTMSpike {

    /*public static void main(String[] args) {
        String libraryPath = System.getProperty("java.library.path");
        System.out.println(libraryPath);
        EagerSession session = EagerSession.create();
        Ops tf = Ops.create(session);
        Scope scope = new Scope(session);

        float[][][] rawInputSequence = {{{0.1f, 0.2f}}, {{0.3f, 0.4f}}}; //shape (timelen, batch_size, num_inputs).
        Operand<TFloat32> inputSequence = tf.constant(rawInputSequence);

        int inputSize = 2;
        int cellSize = 5;
        long maximumTimeLength = 2;

        int[] cellShape = {1, cellSize};
        Operand<TInt32> cellDims = Constant.vectorOf(scope, cellShape);
        Operand<TInt64> seqLenMax = tf.array(maximumTimeLength);
        Operand<TFloat32> initialCellState = Zeros.create(scope, cellDims, TFloat32.DTYPE);
        Operand<TFloat32> initialHiddenState = Zeros.create(scope, cellDims, TFloat32.DTYPE);

        int[] weightShape = {inputSize + cellSize, cellSize * 4};
        Operand<TInt32> weightMatrixDims = Constant.vectorOf(scope, weightShape);
        Operand<TFloat32> weightMatrix = getWeightMatrix(weightMatrixDims, scope);

        int[] weightGatesShape = {cellSize};
        Operand<TInt32> weightGatesDims = Constant.vectorOf(scope, weightGatesShape);
        Operand<TFloat32> weightInputGate = getWeightMatrix(weightGatesDims, scope);
        Operand<TFloat32> weightForgetGate = getWeightMatrix(weightGatesDims, scope);
        Operand<TFloat32> weightOutputGate = getWeightMatrix(weightGatesDims, scope);

        long[] biasShape = {cellSize * 4};
        Operand<TInt64> biasDim = Constant.vectorOf(scope, biasShape);
        Operand<TFloat32> bias = Zeros.create(scope, biasDim, TFloat32.DTYPE);

        BlockLSTM<TFloat32> blockLSTM = BlockLSTM.create(scope, seqLenMax, inputSequence, initialCellState,
                initialHiddenState, weightMatrix, weightInputGate, weightForgetGate, weightOutputGate, bias);

        float[] rawInputGate = TensorValues.castToPrimitiveFloat(blockLSTM.i().asTensor());
        float[] rawCellState = TensorValues.castToPrimitiveFloat(blockLSTM.cs().asTensor());
        float[] rawForgetState = TensorValues.castToPrimitiveFloat(blockLSTM.f().asTensor());
        float[] rawOutputGate = TensorValues.castToPrimitiveFloat(blockLSTM.o().asTensor());
        float[] rawCellInput = TensorValues.castToPrimitiveFloat(blockLSTM.ci().asTensor());
        float[] rawCellOutput = TensorValues.castToPrimitiveFloat(blockLSTM.co().asTensor());
        float[] rawHiddenOutput = TensorValues.castToPrimitiveFloat(blockLSTM.h().asTensor());

        System.out.println("Input Gate: " + Arrays.toString(rawInputGate));
        System.out.println("Cell State: " + Arrays.toString(rawCellState));
        System.out.println("Forget State: " + Arrays.toString(rawForgetState));
        System.out.println("Output Gate: " + Arrays.toString(rawOutputGate));
        System.out.println("Cell Input: " + Arrays.toString(rawCellInput));
        System.out.println("Cell Output: " + Arrays.toString(rawCellOutput));
        System.out.println("Hidden Output: " + Arrays.toString(rawHiddenOutput));

    }

    private static Operand<TFloat32> getWeightMatrix(Operand<TInt32> shape, Scope scope) {
        Tensor<TFloat32> tensorWeight = TensorValues.initializeTruncatedNormalTensor(shape, scope);
        return Constant.create(scope, tensorWeight);
    }*/

}
