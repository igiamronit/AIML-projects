import random
from faker import Faker
import pandas as pd

faker = Faker() #creating faker instance

DEGREES = ["B.Tech", "M.Tech", "BSc", "MSc"] #sample degrees
FIELDS = ["Computer Science", "Electronics", "Information Technology"] #sample fields of study
SKILLS_POOL = ["Python", "Java", "C++", "SQL", "HTML", "CSS", "JavaScript", "React", "Django", "Node.js"] #sample skills

GPT_LINES = [
    "Developed a RESTful API service using Node.js and Express, supporting user authentication and data persistence with MongoDB.",
    "Implemented a responsive front-end dashboard with React and Redux, consuming data from multiple microservices.",
    "Designed and optimized SQL queries for a PostgreSQL database, improving report generation time by 15%.",
    "Contributed to a large-scale Java Spring Boot application, focusing on API development and integration testing.",
    "Built a CI/CD pipeline using Jenkins and Docker for automated deployment of Python web services.",
    "Managed cloud infrastructure on AWS, provisioning EC2 instances, S3 buckets, and RDS databases using Terraform.",
    "Developed mobile application features for iOS using Swift, integrating with Firebase for real-time data.",
    "Wrote high-performance C++ modules for a real-time trading system, reducing execution latency.",
    "Created data ingestion pipelines with Apache Kafka and Spark, processing large volumes of streaming data.",
    "Led the migration of a monolithic application to a microservices architecture using Kubernetes.",
    "Developed a machine learning model in Python using TensorFlow, achieving 92% accuracy on a classification task.",
    "Implemented secure authentication flows using OAuth 2.0 and JWT in a Golang backend service.",
    "Optimized Ruby on Rails application performance by refactoring ActiveRecord queries and caching strategies.",
    "Designed and implemented a serverless function in AWS Lambda using Python for image processing.",
    "Fixed critical bugs and improved error handling in a legacy Perl script for data synchronization.",
    "Collaborated on an Android application development using Kotlin, improving UI responsiveness.",
    "Authored comprehensive unit and integration tests with Jest and React Testing Library for a critical module.",
    "Managed version control with Git and GitHub, conducting code reviews for junior developers.",
    "Deployed containerized applications to Azure Kubernetes Service (AKS), ensuring high availability.",
    "Developed a GraphQL API with Apollo Server and Node.js for a new content management system.",
    "Improved database schema design and indexing for a MySQL database, boosting query performance.",
    "Integrated third-party APIs (Stripe, Twilio) into an e-commerce platform built with Django.",
    "Managed and troubleshooted Kafka clusters, ensuring reliable message delivery for analytics pipelines.",
    "Implemented real-time chat functionality using WebSockets and Node.js for a social networking app.",
    "Developed custom WordPress plugins with PHP, enhancing website functionality and user experience.",
    "Refactored a large JavaScript codebase to TypeScript, improving type safety and maintainability.",
    "Configured and maintained Prometheus and Grafana for system monitoring and alerting.",
    "Automated infrastructure provisioning using Ansible playbooks for server setup and configuration.",
    "Built a desktop application with Electron and React, providing cross-platform compatibility.",
    "Designed and implemented a distributed caching layer using Redis for frequently accessed data.",
    "Developed algorithms for optimal resource allocation in a cloud computing environment using Python.",
    "Contributed to open-source project by submitting pull requests and reviewing code in Rust.",
    "Managed Git repositories and implemented branching strategies (Gitflow) for team collaboration.",
    "Implemented secure REST endpoints and data encryption for a healthcare application.",
    "Conducted performance profiling and identified bottlenecks in a Java application, recommending optimizations.",
    "Built a scalable data warehousing solution using Amazon Redshift and ETL processes.",
    "Developed interactive data visualizations using D3.js for business intelligence dashboards.",
    "Configured Nginx as a reverse proxy and load balancer for microservices deployments.",
    "Assisted in the transition from SVN to Git, providing training and support to the development team.",
    "Designed and implemented a robust error logging and monitoring system using ELK stack (Elasticsearch, Logstash, Kibana).",
    "Developed command-line tools in Go for internal development workflows and automation.",
    "Implemented server-side rendering (SSR) for a Next.js application, improving SEO and initial load times.",
    "Optimized network configurations and firewall rules for cloud-based applications on GCP.",
    "Built a secure payment gateway integration using PCI DSS compliance standards.",
    "Developed embedded software in C for an IoT device, managing sensor data acquisition.",
    "Created comprehensive API documentation using Swagger/OpenAPI for developer consumption.",
    "Improved build times for a large JavaScript project using Webpack optimizations.",
    "Managed and scaled Docker Swarm clusters for container orchestration.",
    "Contributed to the development of a real-time analytics dashboard with Apache Flink.",
    "Implemented A/B testing frameworks within a web application to optimize user engagement.",
    "Developed data validation and cleansing scripts in Python for a data migration project.",
    "Created serverless APIs using Azure Functions and C# for event-driven processing.",
    "Integrated third-party authentication providers (Google, Facebook) into a web application.",
    "Performed security audits and penetration testing on web applications to identify vulnerabilities.",
    "Designed and developed a user role and permission management system for an enterprise application.",
    "Implemented message queues with RabbitMQ for asynchronous task processing.",
    "Built a cross-platform mobile app using React Native, delivering a consistent user experience.",
    "Developed custom shaders and rendering pipelines in Unity for a 3D simulation.",
    "Managed relational and NoSQL databases (Cassandra, Redis) in production environments.",
    "Implemented robust data backup and recovery strategies for critical application data.",
    "Developed a recommendation engine using collaborative filtering techniques in Python.",
    "Configured and managed virtual private clouds (VPCs) and subnets in AWS for network isolation.",
    "Automated software releases using Git tags and release management tools.",
    "Participated in agile ceremonies (scrum, sprint planning, retrospectives) for project delivery.",
    "Developed a custom analytics dashboard using Power BI, integrating with various data sources.",
    "Implemented robust data validation rules and sanitization for user input in a web form.",
    "Integrated Salesforce API for CRM data synchronization in an enterprise application.",
    "Optimized front-end asset loading using CDN and image optimization techniques.",
    "Built a secure file upload service using AWS S3 and pre-signed URLs.",
    "Developed real-time bidding algorithms in Scala for an ad-tech platform.",
    "Managed and optimized Elasticsearch clusters for full-text search capabilities.",
    "Implemented distributed tracing with OpenTelemetry for microservices monitoring.",
    "Developed a desktop application using C# and WPF for internal tooling.",
    "Contributed to the design and implementation of a new database indexing strategy.",
    "Automated testing of API endpoints using Postman and Newman for regression testing.",
    "Developed a browser extension with JavaScript to enhance user productivity.",
    "Managed package dependencies and build processes with npm and Yarn.",
    "Implemented server-side logging with Log4j and centralized logging with Splunk.",
    "Configured Continuous Deployment with GitLab CI for multi-stage deployments.",
    "Developed a simulation environment in Python for testing robotic control algorithms.",
    "Designed and implemented a robust error reporting mechanism with Sentry.",
    "Built a web crawler using Python and Scrapy to collect publicly available data.",
    "Configured cloud firewalls and security groups to protect application endpoints.",
    "Developed custom middleware for an Express.js application to handle request parsing.",
    "Implemented serverless functions using Google Cloud Functions for event-driven tasks.",
    "Managed and maintained Linux servers, ensuring system uptime and security.",
    "Developed a component library with Storybook and React for consistent UI development.",
    "Integrated payment gateways (PayPal, Square) into a customer-facing web portal.",
    "Optimized image delivery using WebP format and responsive image techniques.",
    "Built a data visualization tool using Python and Plotly Dash for internal reporting.",
    "Developed micro-frontends with Web Components for a modular application architecture.",
    "Managed secrets and environment variables using AWS Secrets Manager.",
    "Implemented rate limiting and API throttling for public-facing endpoints.",
    "Developed a desktop application using Java Swing for inventory management.",
    "Integrated push notifications (FCM, APNs) into a mobile application.",
    "Contributed to a high-performance message broker in Go, handling millions of messages daily.",
    "Developed a custom authentication service using Auth0 for multi-tenant applications.",
    "Implemented end-to-end encryption for data at rest and in transit using TLS.",
    "Configured and managed Kubernetes Ingress controllers for external access.",
    "Built a real-time dashboard using Vue.js and WebSockets to display sensor data.",
    "Optimized database connection pooling and transaction management in a Java application.",
    "Developed a command-line interface (CLI) tool in Python for cloud resource management.",
    "Implemented robust caching strategies using Memcached for frequently accessed data.",
    "Contributed to a game engine development in C++ focusing on rendering optimizations.",
    "Managed and maintained Helm charts for Kubernetes application deployments.",
    "Developed custom API gateways with Kong for microservices orchestration and security.",
    "Implemented a data migration script from SQL Server to PostgreSQL using Python.",
    "Built a Progressive Web App (PWA) with Service Workers for offline capabilities.",
    "Configured load balancers and auto-scaling groups for high-traffic web applications.",
    "Developed a chatbot using natural language processing (NLP) with spaCy and Python."
]

def generate_resume(name, degree, skills, years_exp):
    project_lines = random.sample(GPT_LINES, k=2)  # choose 2 random lines
    intro = f"{name} is a {degree} graduate with {years_exp} years of experience in software development."
    skill_line = f"They are proficient in technologies such as {', '.join(skills)}."
    experience = " ".join(f"- {line}" for line in project_lines)
    
    return f"{intro} {skill_line} Work Highlights: {experience}"

#to check if candidate is shortlised or not
def label_candidate(gpa, years_exp, skills):
    good_skills = {"Python", "SQL", "Java", "JavaScript"}
    skill_score = len(set(skills) & good_skills) #intersecting skills with good skills
    if gpa >= 8 and years_exp >= 2 and skill_score >= 2:
        return 1
    elif gpa >= 7 and skill_score >= 3:
        return 1
    return 0

def generate_candidate():
    name = faker.name()
    degree = random.choice(DEGREES)
    field = random.choice(FIELDS)
    gpa = round(random.uniform(6, 10), 2)
    years_exp = round(random.uniform(0, 6), 1)
    age = random.randint(21, 35)
    skills = random.sample(SKILLS_POOL, k=random.randint(3, 6))
    resume = generate_resume(name, degree, skills, years_exp)
    label = label_candidate(gpa, years_exp, skills)

    return {
        "name": name,
        "resume": resume,
        "degree": degree,
        "field_of_study": field,
        "GPA": gpa,
        "years_of_experience": years_exp,
        "age": age,
        #"skills": ", ".join(skills),
        "shortlisted": label
    }
data = [generate_candidate() for _ in range(1000)]
df = pd.DataFrame(data)
df.to_csv("software_engineer_dataset.csv", index=False)