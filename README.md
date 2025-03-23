# Project Title

[[_TOC_]]

## Team Members

1. Mike Boss
2. Konstantin Heep
3. Madeleine Soukup
4. Romeo Stoll

## Project Description

Describe here your project in detail and define your goals.

### Users

List your projects target Users.

### Tasks

Define all the tasks you want your dashboard solve.

---

## Folder Structure

Specify here the structure of you code and comment what the most important files contain

```bash
├── README.md
├── backend
│   ├── setup.py   # main app
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── MANIFEST.in
│   ├── README.md
│   ├── pyproject.toml
│   ├── data
│   │   ├── ames-housing-features.json
│   │   ├── ames-housing-gam-instance-data.json
│   │   └── ames-housing-gam.json
│   └── src/gamut_server
│       ├── resources
│       │   ├── __init__.py
│       │   ├── description.py
│       │   ├── features.py
│       │   └── instances.py
│       ├── router
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── routes.py
│       └── __init__.py
├── frontend
│   ├── README.md
│   ├── package-lock.json
│   ├── package.json
│   ├── src
│   │   ├── App.css
│   │   ├── App.test.tsx
│   │   ├── App.tsx
│   │   ├── Visualization.tsx
│   │   ├── backend
│   │   │   ├── BackendQueryEngine.tsx
│   │   │   └── json-decoder.ts
│   │   ├── components
│   │   │   ├── BasicLineChart
│   │   │   │   ├── BasicLineChart.scss
│   │   │   │   ├── BasicLineChart.tsx
│   │   │   │   └── types.ts
│   │   │   ├── DataChoiceComponent.tsx
│   │   │   ├── DataPointComponent.tsx
│   │   │   └── ScatterPlot
│   │   │       ├── ScatterPlot.scss
│   │   │       ├── ScatterPlot.tsx
│   │   │       └── types.ts
│   │   ├── index.css
│   │   ├── index.tsx
│   │   ├── logo.svg
│   │   ├── react-app-env.d.ts
│   │   ├── reportWebVitals.ts
│   │   ├── setupTests.ts
│   │   └── types
│   │       ├── DataArray.ts
│   │       ├── DataPoint.ts
│   │       └── Margins.ts
│   └── tsconfig.json
└── requirements.txt
```

## Requirements

Write here all intructions to build the environment and run your code.\
**NOTE:** If we cannot run your code following these requirements we will not be able to evaluate it.

## How to Run

### Using Docker Compose (Recommended)

To run the development server using Docker Compose:

1. Make sure you have Docker and Docker Compose installed
2. From the project root directory, run:
   ```bash
   docker compose up
   ```
   This will start both the frontend and backend services in development mode.
   - Frontend will be available at: http://localhost:3000
   - Backend will be available at: http://localhost:8080/api/v1/

To rebuild the containers (e.g., after making changes to the Dockerfile or dependencies):

```bash
docker compose up --build
```

To view the logs:

```bash
docker compose logs -f
```

You can also view logs for specific services:

```bash
docker compose logs -f frontend  # for frontend logs
docker compose logs -f backend   # for backend logs
```

To stop the services:

```bash
docker compose down
```

## Milestones

Document here the major milestones of your code and future planned steps.\

- [x] Week 1

  - [x] Completed Sub-task: [#20984ec2](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/dummy-fullstack/-/commit/20984ec2197fa8dcdc50f19723e5aa234b9588a3)
  - [x] Completed Sub-task: ...

- [ ] Week 2
  - [ ] Sub-task: [#2](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/dummy-fullstack/-/issues/2)
  - [ ] Sub-task: ...

Create a list subtask.\
Open an issue for each subtask. Once you create a subtask, link the corresponding issue.\
Create a merge request (with corresponding branch) from each issue.\
Finally accept the merge request once issue is resolved. Once you complete a task, link the corresponding merge commit.\
Take a look at [Issues and Branches](https://www.youtube.com/watch?v=DSuSBuVYpys) for more details.

This will help you have a clearer overview of what you are currently doing, track your progress and organise your work among yourselves. Moreover it gives us more insights on your progress.

## Weekly Summary

Write here a short summary with weekly progress, including challanges and open questions.\
We will use this to understand what your struggles and where did the weekly effort go to.

## Versioning

Create stable versions of your code each week by using gitlab tags.\
Take a look at [Gitlab Tags](https://docs.gitlab.com/ee/topics/git/tags.html) for more details.

Then list here the weekly tags. \
We will evaluate your code every week, based on the corresponding version.

Tags:

- Week 1: [Week 1 Tag](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/dummy-fullstack/-/tags/stable-readme)
- Week 2: ..
- Week 3: ..
- ...

link pitch video:
https://polybox.ethz.ch/index.php/s/bjtuIzQxFkGUOqX
