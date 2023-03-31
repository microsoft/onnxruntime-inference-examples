package ai.onnxruntime.example.question_answering

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.util.*


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private val TitleText: TextView by lazy {findViewById(R.id.TitleView)}
    private val ArticleText: TextView by lazy {findViewById(R.id.ArticleView)}
    private val AnswerText: TextView by lazy {findViewById(R.id.AnswerView)}
    private val QuestionText: TextView by lazy {findViewById(R.id.QuestionView)}
    private val DoQAButton: Button by lazy {findViewById(R.id.PerformQaButton)}

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // from <The Little Prince>, chapter one, and the first paragraph
        TitleText.setText("<The Little Prince>")
        ArticleText.setText(
            "we are introduced to the narrator, a pilot, and his ideas about grown-ups." +
                    "Once when I was six years old I saw a magnificent picture in a book," +
                    " called True Stories from Nature, about the primeval forest. " +
                    "It was a picture of a boa constrictor in the act of swallowing an animal. " +
                    "Here is a copy of the drawing." +
                    "In the book it said: \"Boa constrictors swallow their prey whole, " +
                    "without chewing it. After that they are not able to move," +
                    " and they sleep through the six months that they need for digestion.\""
        );
        ArticleText.addTextChangedListener(object:TextWatcher {
            override fun beforeTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {

            }

            override fun onTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {

            }

            override fun afterTextChanged(s: Editable) {
                TitleText.setText("Customized context")
            }

        })
        QuestionText.setHint("From which book did I see a magnificent picture?")

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to load tokenizer and post-processor ops
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        DoQAButton.setOnClickListener {
            try {
                performQA(ortSession)
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform QuestionAnswering", e)
                Toast.makeText(
                    baseContext,
                    "Failed to perform QuestionAnswering",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun updateUI(result: Result) {
        val default_ans: String = "No answer found."
        AnswerText.setText(
            if (result.outputAnswer.equals("[CLS]")) default_ans else result.outputAnswer
        )
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.csarron_mobilebert_uncased_squad_v2_quant_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readQuestion(): CharSequence {
        var user_text = QuestionText.text
        return if (user_text.isEmpty() == true) QuestionText.hint else user_text
    }

    private fun performQA(ortSession: OrtSession) {
        var qaPerformer = QAPerformer()
        var result = qaPerformer.answer(ArticleText.text, readQuestion(), ortEnv, ortSession)
        updateUI(result);
    }

    companion object {
        const val TAG = "ORTQuestionAnswering"
    }
}
