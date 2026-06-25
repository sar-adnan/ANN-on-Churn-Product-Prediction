from pydantic import BaseModel, Field, validator
from typing import Optional
import pandas as pd

class CustomerData(BaseModel):
    gender: int = Field(..., ge=0, le=1)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    tenure: float = Field(..., ge=0.0, le=1.0)
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=2)
    InternetService: int = Field(..., ge=0, le=2)
    OnlineSecurity: int = Field(..., ge=0, le=2)
    OnlineBackup: int = Field(..., ge=0, le=2)
    DeviceProtection: int = Field(..., ge=0, le=2)
    TechSupport: int = Field(..., ge=0, le=2)
    StreamingTV: int = Field(..., ge=0, le=2)
    StreamingMovies: int = Field(..., ge=0, le=2)
    Contract: int = Field(..., ge=0, le=2)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    PaymentMethod: int = Field(..., ge=0, le=3)
    MonthlyCharges: float = Field(..., ge=0.0, le=1.0)
    TotalCharges: float = Field(..., ge=0.0, le=1.0)

def validate_dataframe(df: pd.DataFrame):
    """
    Validates a pandas DataFrame against the CustomerData schema.
    """
    errors = []
    for index, row in df.iterrows():
        try:
            CustomerData(**row.to_dict())
        except Exception as e:
            errors.append(f"Row {index}: {str(e)}")
    
    if errors:
        print(f"Validation failed with {len(errors)} errors.")
        # In a production scenario, you might raise an exception or log these errors.
        return False
    else:
        print("Data validation successful.")
        return True

if __name__ == "__main__":
    # Example usage for testing
    print("Testing data validation...")
    # This is just a placeholder to demonstrate the script's functionality.
    # In practice, this would be called from the data preprocessing or training pipeline.
