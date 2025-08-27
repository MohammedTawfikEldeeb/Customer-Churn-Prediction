from pydantic import BaseModel , Field


class ProcessReuest(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    UsageFrequency: float = Field(..., alias="Usage Frequency")
    SupportCalls: float = Field(..., alias="Support Calls")
    PaymentDelay: float = Field(..., alias="Payment Delay")
    SubscriptionType: str = Field(..., alias="Subscription Type")
    ContractLength: str = Field(..., alias="Contract Length")
    TotalSpent: float = Field(..., alias="Total Spent")
    LastInteraction: float = Field(..., alias="Last Interaction")





class SimpleChurnResponse(BaseModel):
    churn_prediction: int  