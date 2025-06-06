# utils/preprocessing.py
import numpy as np
import pandas as pd

def preprocess_general_input_data(data_dict, all_features_ordered, numeric_features_to_scale,
                                scaler, categorical_features=None, le_dict=None,
                                binary_features=None, special_handling=None):
    try:
        # Special handling of the input data dictionary if provided
        if special_handling:
            data_dict = special_handling(data_dict.copy())

        # Convert input dictionary to DataFrame and reindex to ensure correct column order and presence
        input_df = pd.DataFrame([data_dict])
        input_df = input_df.reindex(columns=all_features_ordered)

        # Check for missing features in the initial payload
        expected_input_features = [
            col for col in all_features_ordered
            if not (special_handling and col == 'bmi' and 'bmi' not in data_dict)
        ]
        missing_cols_in_payload = [col for col in expected_input_features if col not in data_dict]
        if missing_cols_in_payload:
            raise ValueError(f"Fitur hilang dalam payload JSON: {missing_cols_in_payload}.")

        # Check for initial NaN values in features that should be present
        cols_to_check_nan_initially = [
            col for col in input_df.columns
            if not (special_handling and col == 'bmi' and col not in data_dict)
        ]
        if input_df[cols_to_check_nan_initially].isnull().any().any():
            nan_cols = input_df[cols_to_check_nan_initially].columns[input_df[cols_to_check_nan_initially].isnull().any()].tolist()
            truly_null_in_input = [col for col in nan_cols if col in data_dict and pd.isna(data_dict[col])]
            if truly_null_in_input:
                raise ValueError(f"Fitur berikut tidak boleh bernilai null dalam input: {truly_null_in_input}.")

        # Handle categorical features using Label Encoding
        if categorical_features and le_dict:
            for col in categorical_features:
                if col in input_df.columns:
                    if pd.isna(input_df[col].iloc[0]):
                        raise ValueError(f"Fitur kategorikal '{col}' tidak boleh kosong.")
                    le = le_dict.get(col)
                    if le:
                        try:
                            input_value = input_df[col].iloc[0]
                            input_value_array = [input_value] if not isinstance(input_value, (list, np.ndarray, pd.Series)) else input_value

                            if not all(item in le.classes_ for item in input_value_array):
                                unknown_values = [item for item in input_value_array if item not in le.classes_]
                                raise ValueError(f"Nilai tidak dikenal untuk fitur '{col}': {unknown_values}. Nilai yang diketahui: {list(le.classes_)}")

                            input_df[col] = le.transform(input_value_array)
                        except ValueError as e_transform:
                            original_value = data_dict.get(col, "N/A")
                            raise ValueError(f"Error encoding fitur '{col}' dengan nilai '{original_value}'. Detail: {str(e_transform)}")
                    else:
                        raise ValueError(f"LabelEncoder tidak ditemukan untuk fitur: {col}")

        # Handle binary features (ensure they are numeric 0 or 1)
        if binary_features:
            for col_bin in binary_features:
                if col_bin in input_df.columns:
                    if pd.isna(input_df[col_bin].iloc[0]):
                        raise ValueError(f"Fitur biner '{col_bin}' tidak boleh kosong.")
                    try:
                        val = pd.to_numeric(input_df[col_bin].iloc[0])
                        if val not in [0, 1] and col_bin in ['hypertension', 'heart_disease']:
                            raise ValueError(f"Fitur biner '{col_bin}' hanya boleh 0 atau 1, diterima: {val}")
                        input_df[col_bin] = val
                    except ValueError as e_bin:
                        raise ValueError(f"Fitur biner '{col_bin}' harus berupa angka (0 atau 1). Error: {str(e_bin)}")

        # Scale numeric features
        if numeric_features_to_scale and scaler:
            nan_in_numeric_cols = input_df[numeric_features_to_scale].isnull().any()
            if nan_in_numeric_cols.any():
                cols_with_nan = nan_in_numeric_cols[nan_in_numeric_cols].index.tolist()
                raise ValueError(f"Fitur numerik berikut tidak boleh kosong sebelum scaling: {cols_with_nan}")
            try:
                for col_num in numeric_features_to_scale:
                    input_df[col_num] = pd.to_numeric(input_df[col_num], errors='raise')
                input_df[numeric_features_to_scale] = scaler.transform(input_df[numeric_features_to_scale])
            except ValueError as ve:
                raise ValueError(f"Error saat scaling fitur numerik. Pastikan semuanya numerik dan tidak ada nilai yang hilang. Detail: {str(ve)}")
            except Exception as e:
                raise RuntimeError(f"Error tidak terduga saat scaling: {str(e)}")
        elif numeric_features_to_scale and not scaler:
            raise ValueError("Scaler dibutuhkan untuk fitur numerik tetapi tidak tersedia/gagal dimuat.")

        return input_df

    except KeyError as e:
        raise ValueError(f"Fitur hilang dalam struktur data internal (KeyError): {str(e)}. Ini mungkin bug.")
    except ValueError as e_val:
        raise e_val
    except Exception as e_gen:
        print(f"Unexpected error in preprocessing: {e_gen}")
        raise RuntimeError(f"Terjadi kesalahan tidak terduga saat memproses data: {str(e_gen)}")


def handle_cardiovascular_features(data_dict):
    """Menghitung BMI dan menambahkannya ke data_dict untuk model kardiovaskular."""
    try:
        if 'height' not in data_dict or 'weight' not in data_dict:
            raise ValueError("Fitur 'height' dan 'weight' dibutuhkan untuk menghitung BMI.")
        
        height_cm_orig = data_dict['height']
        weight_kg_orig = data_dict['weight']

        if pd.isna(height_cm_orig) or pd.isna(weight_kg_orig):
            raise ValueError("Nilai 'height' dan 'weight' tidak boleh kosong untuk menghitung BMI.")

        height_cm = pd.to_numeric(height_cm_orig, errors='coerce')
        weight_kg = pd.to_numeric(weight_kg_orig, errors='coerce')

        if pd.isna(height_cm) or pd.isna(weight_kg):
            err_msg = []
            if pd.isna(height_cm): err_msg.append(f"'height' ('{height_cm_orig}')")
            if pd.isna(weight_kg): err_msg.append(f"'weight' ('{weight_kg_orig}')")
            raise ValueError(f"Fitur berikut harus berupa angka untuk menghitung BMI: {', '.join(err_msg)}.")

        if height_cm <= 0:
            raise ValueError(f"Tinggi badan ('height': {height_cm}) harus lebih besar dari 0 untuk menghitung BMI.")
        if weight_kg <= 0:
            raise ValueError(f"Berat badan ('weight': {weight_kg}) harus lebih besar dari 0 untuk menghitung BMI.")
            
        data_dict['bmi'] = weight_kg / (height_cm / 100)**2
    except ValueError as e: # Menangkap error dari validasi di atas
        raise ValueError(f"Input tidak valid untuk perhitungan BMI: {str(e)}")
    except Exception as e: # Menangkap error lain yang mungkin terjadi
        raise ValueError(f"Error saat menghitung BMI: {str(e)}")
    return data_dict