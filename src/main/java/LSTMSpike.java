import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Zeros;
import org.tensorflow.op.nn.BlockLSTM;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class LSTMSpike {

    public static void main(String[] args) {
        try (EagerSession session = EagerSession.create()){
            Ops tf = Ops.create(session);
            Scope scope = new Scope(session);

            float[][][] rawInputSequence = {{{0.1f, 0.2f}}, {{0.3f, 0.4f}}}; //shape (timelen, batch_size, num_inputs).
            Operand<TFloat32> inputSequence = tf.constant(rawInputSequence);

            int inputSize = 2;
            long cellSize = 5;
            long maximumTimeLength = 2;

            long[] cellShape = {1, cellSize};
            Operand<TInt64> cellDims = Constant.vectorOf(scope, cellShape);
            Operand<TInt64> seqLenMax = tf.array(maximumTimeLength);
            Operand<TFloat32> initialCellState = Zeros.create(scope, cellDims, TFloat32.DTYPE);
            Operand<TFloat32> initialHiddenState = Zeros.create(scope, cellDims, TFloat32.DTYPE);

            long[] weightShape = {inputSize + cellSize, cellSize * 4};
            Operand<TInt64> weightMatrixDims = Constant.vectorOf(scope, weightShape);
            Operand<TFloat32> weightMatrix = Zeros.create(scope, weightMatrixDims, TFloat32.DTYPE);

            long[] weightGatesShape = {cellSize};
            Operand<TInt64> weightGatesDims = Constant.vectorOf(scope, weightGatesShape);
            Operand<TFloat32> weightInputGate = Zeros.create(scope, weightGatesDims, TFloat32.DTYPE);
            Operand<TFloat32> weightForgetGate = Zeros.create(scope, weightGatesDims, TFloat32.DTYPE);
            Operand<TFloat32> weightOutputGate = Zeros.create(scope, weightGatesDims, TFloat32.DTYPE);

            long[] biasShape = {cellSize * 4};
            Operand<TInt64> biasDim = Constant.vectorOf(scope, biasShape);
            Operand<TFloat32> bias = Zeros.create(scope, biasDim, TFloat32.DTYPE);

            BlockLSTM<TFloat32> blockLSTM = BlockLSTM.create(scope, seqLenMax, inputSequence, initialCellState,
                    initialHiddenState, weightMatrix, weightInputGate, weightForgetGate, weightOutputGate, bias);

            float[] rawInputGate = getTensorValues(blockLSTM.i().asTensor());
            float[] rawCellState = getTensorValues(blockLSTM.cs().asTensor());
            float[] rawForgetState = getTensorValues(blockLSTM.f().asTensor());
            float[] rawOutputGate = getTensorValues(blockLSTM.o().asTensor());
            float[] rawCellInput = getTensorValues(blockLSTM.ci().asTensor());
            float[] rawCellOutput = getTensorValues(blockLSTM.co().asTensor());
            float[] rawHiddenOutput = getTensorValues(blockLSTM.h().asTensor());

            System.out.println("Input Gate: "+ Arrays.toString(rawInputGate));
            System.out.println("Cell State: "+ Arrays.toString(rawCellState));
            System.out.println("Forget State: "+ Arrays.toString(rawForgetState));
            System.out.println("Output Gate: "+ Arrays.toString(rawOutputGate));
            System.out.println("Cell Input: "+ Arrays.toString(rawCellInput));
            System.out.println("Cell Output: "+ Arrays.toString(rawCellOutput));
            System.out.println("Hidden Output: "+ Arrays.toString(rawHiddenOutput));

        }
    }

    private static float[] getTensorValues(Tensor<TFloat32> tensor) {
        FloatBuffer foreignBuffer = ByteBuffer.allocate((int) tensor.numBytes())
                .order(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                        ? ByteOrder.BIG_ENDIAN
                        : ByteOrder.LITTLE_ENDIAN).asFloatBuffer();

        tensor.rawData().asFloats().copyTo(DataBuffers.of(foreignBuffer), foreignBuffer.capacity());
        float[] tensorValues = new float[foreignBuffer.remaining()];
        foreignBuffer.get(tensorValues);
        foreignBuffer.clear();

        return tensorValues;
    }

}
