package ai.onnxruntime.genai.demo;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.google.android.material.bottomsheet.BottomSheetDialogFragment;

public class BottomSheet extends BottomSheetDialogFragment {
    private EditText maxLengthEditText;
    private EditText lengthPenaltyEditText;
    private SettingsListener settingsListener;

    public interface SettingsListener {
        void onSettingsApplied(int maxLength, float lengthPenalty);
    }

    public void setSettingsListener(SettingsListener listener) {
        this.settingsListener = listener;
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.bottom_sheet, container, false);

        maxLengthEditText = view.findViewById(R.id.idEdtMaxLength);
        lengthPenaltyEditText = view.findViewById(R.id.idEdtLengthPenalty);

        Button applyButton = view.findViewById(R.id.applySettingsButton);

        applyButton.setOnClickListener(v -> {
            if (settingsListener != null) {
                int maxLength = Integer.parseInt(maxLengthEditText.getText().toString());
                float lengthPenalty = Float.parseFloat(lengthPenaltyEditText.getText().toString());
                settingsListener.onSettingsApplied(maxLength, lengthPenalty);
                dismiss();
            }
        });

        return view;
    }
}
