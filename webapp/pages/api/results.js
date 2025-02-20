import prisma from "../../prisma/client";

export default async function handler(req, res) {
  if (req.method === 'POST') {

    if (!req.body.jobId || !req.body.status || (req.body.status !== 'completed' || req.body.status !== 'failed') || !req.body.elapsed) {
      res.status(400)
    }

    await prisma.result.create({
      data: {
        jobId: req.body.jobId,

        f1: req.body.f1,
        precision: req.body.precision,
        recall: req.body.recall,
        trainTime: req.body.trainTime,
        evalTime: req.body.evalTime,

        cpuUtilized: req.body.cpuUtilized,
        memoryUtilized: req.body.memoryUtilized,
        gpuAllocated: req.body.gpuAllocated,
        gpuUtilized: req.body.gpuUtilized,
        gpuMemUtilized: req.body.gpuMemUtilized,
        energyConsumed: req.body.energyConsumed,
        totalRuntime: req.body.totalRuntime,
      }
    });

    await prisma.job.update({
      where: {
        id: req.body.jobId
      },
      data: {
        status: req.body.status
      }
    });

    res.status(200);
  } else {
    res.status(405)
  }
}
