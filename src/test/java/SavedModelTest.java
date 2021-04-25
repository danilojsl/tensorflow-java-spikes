import org.junit.Test;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.DataType;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TUint8;
import util.TensorValues;

import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotNull;

public class SavedModelTest {

    private static final String SAVED_MODEL_PATH;
    private static final String SAVED_MODEL_DP_PATH;
    private static final String SAVED_MODEL_PY_PATH;

    static {
        try {
            SAVED_MODEL_PATH = Paths.get(SavedModelTest.class.getResource("/saved_model").toURI()).toString();
            SAVED_MODEL_DP_PATH = Paths.get(SavedModelTest.class.getResource("/saved_dp_model/BiLSTM").toURI()).toString();
            SAVED_MODEL_PY_PATH = Paths.get(SavedModelTest.class.getResource("/saved_model_using_python/model").toURI()).toString();
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void load() {
        try (SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PATH, SavedModelBundle.DEFAULT_TAG)) {
            assertNotNull(bundle.session());
            assertNotNull(bundle.graph());
            assertNotNull(bundle.metaGraphDef());
        }
    }

    @Test
    public void loadParser() {
        try (SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_DP_PATH, SavedModelBundle.DEFAULT_TAG)) {
            assertNotNull(bundle.session());
            assertNotNull(bundle.graph());
            assertNotNull(bundle.metaGraphDef());
        }
    }

    @Test
    public void loadFunctions() {
        SavedModelBundle savedModel = SavedModelBundle.load(SAVED_MODEL_DP_PATH, SavedModelBundle.DEFAULT_TAG);
        ConcreteFunction function = savedModel.function(Signature.DEFAULT_KEY);
        Signature signature = function.signature();
        assertNotNull(signature);

    }

    @Test
    public void pythonTfFunction() {
        // ConcreteFunctions on models saved using python
        try (SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PY_PATH, "serve")) {
            /*
             * Test model was created in python
             *   Signature name used for saving 'add', argument names 'a' and 'b'
             */
            ConcreteFunction add = bundle.function("add");
            Map<String, Tensor<?>> args = new HashMap();
            try (Tensor<TFloat32> a = TFloat32.scalarOf(10.0f);
                 Tensor<TFloat32> b = TFloat32.scalarOf(15.5f)) {
                args.put("a", a);
                args.put("b", b);
                Map<String, Tensor<?>> result = add.call(args);
                assertEquals(result.size(), 1);
                try (Tensor<TFloat32> c = result.values().iterator().next().expect(TFloat32.DTYPE)) {
                    assertEquals(25.5f, c.data().getFloat());
                }
            }
        }
    }

    @Test
    public void pythonVectorFunction() {
        // ConcreteFunctions on models saved using python
        try (SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PY_PATH, "serve")) {
            /*
             * Test model was created in python
             *   Signature name used for saving 'add', argument names 'a' and 'b'
             */
            ConcreteFunction getVector = bundle.function("get_const_vector");
            Map<String, Tensor<?>> args = new HashMap();
            Tensor<TFloat32> input = TFloat32.scalarOf(10.0f);
            args.put("input", input);
            Map<String, Tensor<?>> result = getVector.call(args);
            Tensor<TFloat32> output = result.get("output_0").expect(TFloat32.DTYPE);
            float[] rawOutput = TensorValues.castToPrimitiveFloat(output);
            System.out.println(Arrays.toString(rawOutput));
        }
    }

    @Test
    public void pythonSomeFunction() {
        // ConcreteFunctions on models saved using python
        SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PY_PATH, "serve");
        /*
         * Test model was created in python
         *   Signature name used for saving 'add', argument names 'a' and 'b'
         */
        ConcreteFunction getVector = bundle.function("some_function");
        Map<String, Tensor<?>> args = new HashMap();
        Tensor<TFloat32> x = TFloat32.vectorOf(2, 3);
        Tensor<TFloat32> y = TFloat32.vectorOf(3, -2);
        args.put("x", x);
        args.put("y", y);
        Map<String, Tensor<?>> result = getVector.call(args);
        Tensor<TFloat32> output = result.get("output_0").expect(TFloat32.DTYPE);
        float[] rawOutput = TensorValues.castToPrimitiveFloat(output);
        System.out.println(Arrays.toString(rawOutput));
    }


    @Test
    public void pythonGlorotOutputFunction() {
        // ConcreteFunctions on models saved using python
        SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PY_PATH, "serve");

        ConcreteFunction getGlorotInitializer = bundle.function("glorot_initializer");
        Tensor<TInt32> dummy = TInt32.scalarOf(1);
        Map<String, Tensor<?>> args = new HashMap();
        args.put("dummy", dummy);
        Map<String, Tensor<?>> result = getGlorotInitializer.call(args);
        Tensor<TFloat32> output = result.get("output_0").expect(TFloat32.DTYPE);
        float[] rawOutput = TensorValues.castToPrimitiveFloat(output);
        System.out.println(Arrays.toString(rawOutput));
    }

    @Test
    public void pythonLstmOutputFunction() {
        // ConcreteFunctions on models saved using python
        SavedModelBundle bundle = SavedModelBundle.load(SAVED_MODEL_PY_PATH, "serve");

        ConcreteFunction getLstmOutput = bundle.function("lstm_output");
        //float [][]
        Map<String, Tensor<?>> args = new HashMap();

    }

    @Test
    public void shouldRestoreVariablesFromSavedModel() {
        SavedModelBundle model = SavedModelBundle.load(SAVED_MODEL_DP_PATH, SavedModelBundle.DEFAULT_TAG);
        ConcreteFunction concreteFunction = model.function("serving_default");
        MetaGraphDef metaGraphDef = model.metaGraphDef();
        SignatureDef sig = metaGraphDef.getSignatureDefOrThrow("serving_default");
        System.out.println("model");
        //Map<String, Tensor> feedDict = new HashMap<>();
        //feedDict.put("input_tensor", reshapeTensor);
        //Map<String, Tensor> outputTensorMap = model.function("serving_default").call(feedDict);

    }


}
