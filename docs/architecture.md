# CircuitSentry Architecture

This diagram visualizes the architecture and agentic data flow for the CircuitSentry system.

```mermaid
flowchart TD
    subgraph Local["Local Environment (User's Laptop)"]
        LocalClient["Capture Client<br>(Python + OpenCV)"]
        Webcam["Webcam"]
        Webcam -->|Raw Frames| LocalClient
    end

    subgraph AWSCloud["AWS Cloud Serverless Infrastructure"]
        APIGW["API Gateway<br>(POST /capture)"]
        CaptureLambda["Capture Trigger Lambda"]
        
        subgraph StepFunctions["Step Functions State Machine (Agent Workflow)"]
            Perception["Perception Task<br>(Lambda + OpenCV)"]
            Decision{"Decision Engine<br>(Confidence Check)"}
            Pass["Terminal: Pass"]
            FlagForApproval["Terminal: Flag for Approval"]
            RequestRecapture["Action: Request Re-Capture<br>(Adjust ROI/Exposure)"]
            
            Perception --> Decision
            Decision -->|Confidence < LOW| Pass
            Decision -->|Confidence >= HIGH| FlagForApproval
            Decision -->|LOW <= Confidence < HIGH<br>AND Loop Count <= 2| RequestRecapture
            Decision -->|Loop Count > 2| FlagForApproval
            RequestRecapture -->|Recapture Command| Perception
        end

        DynamoDB[("DynamoDB<br>(Audit Trail / Evaluation Dataset)")]
        S3Bucket[("Amazon S3<br>(Raw & Cropped Images)")]
        
        subgraph DashboardApp["Dashboard (Web UI)"]
            CloudFront["CloudFront"]
            DashboardS3[("S3 Static Site Hosting")]
            DashboardAPIGW["Dashboard API Gateway"]
            
            CloudFront --> DashboardS3
        end
    end
    
    LocalClient -->|1. Upload presigned URL| S3Bucket
    LocalClient -->|2. Trigger request| APIGW
    LocalClient <..>|Polls/Fulfills Recapture Cmds| RequestRecapture
    
    APIGW --> CaptureLambda
    CaptureLambda --> StepFunctions
    
    Perception <-->|Read/Write Image| S3Bucket
    Perception -->|Log Finding| DynamoDB
    FlagForApproval -->|Mark Pending Approval| DynamoDB
    Pass -->|Mark Clean| DynamoDB
    
    CloudFront --> DashboardAPIGW
    DashboardAPIGW -->|Read Runs| DynamoDB
    DashboardAPIGW -->|Read Images| S3Bucket
```

## Flow Description

1. **Local Capture**: The Python capture client reads frames from the webcam. When an inspection is requested, it uploads the frame to S3 and triggers the API Gateway.
2. **Workflow Orchestration**: The API Gateway passes the request to a Lambda function, which kicks off a unique AWS Step Functions execution (the core agent).
3. **Agentic Perception & Decision**: 
    - The **Perception Lambda** (running OpenCV) aligns the image, diffs it against a golden reference, and scores defects.
    - The **Decision Choice State** evaluates the confidence:
        - Definitively clean (`< LOW`) triggers a Pass.
        - Definitively defective (`>= HIGH`) flags it in the database for human review.
        - Ambiguous cases trigger a request for the local client to re-capture the specific region.
4. **Data & Presentation**: All decisions, bounding boxes, and metadata are continuously logged to DynamoDB. The CloudFront-hosted dashboard reads this data (and the corresponding images from S3) to let users view live traces and approve/reject flagged cases.
